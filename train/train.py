from tqdm import trange
import torch.nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tools.utils import *
from tools.ops import compute_grad_gp, update_average, copy_norm_params, queue_data, dequeue_data, \
    average_gradients, calc_adv_loss, calc_contrastive_loss, calc_recon_loss, \
    calc_abl, calc_pseudo_hamming_loss, calc_wdl, calc_pkl
from tools.hsic import RbfHSIC


def add_indp_fact_loss(self, *exp_pairs):
    pairs = []
    for _exp1, _exp2 in exp_pairs:
        _pairs = [(F.adaptive_avg_pool2d(_exp1[:, i], 1).squeeze(),
                    F.adaptive_avg_pool2d(_exp2[:, i], 1).squeeze())
                    for i in range(_exp1.shape[1])]
        pairs += _pairs

    crit = RbfHSIC(1)
    losses = [crit(*pair) for pair in pairs]
    return losses

def trainGAN(data_loader, networks, opts, epoch, args, additional, \
                detach=False, quantize=False, style_con=False, \
                reconstruction_losses=False, hsic_loss=False, abl=False,\
                phl=False, wdl=False):
    # avg meter
    d_losses = AverageMeter()
    d_advs = AverageMeter()
    d_gps = AverageMeter()

    g_losses = AverageMeter()
    g_advs = AverageMeter()
    g_imgrecs = AverageMeter()
    g_rec = AverageMeter()

    moco_losses = AverageMeter()

    # set nets
    D = networks['D'] if not args.distributed else networks['D'].module
    G = networks['G'] if not args.distributed else networks['G'].module
    C = networks['C'] if not args.distributed else networks['C'].module
    G_EMA = networks['G_EMA'] if not args.distributed else networks['G_EMA'].module
    C_EMA = networks['C_EMA'] if not args.distributed else networks['C_EMA'].module
    # set opts
    d_opt = opts['D']
    g_opt = opts['G']
    c_opt = opts['C']
    # switch to train mode
    D.train()
    G.train()
    C.train()
    C_EMA.train()
    G_EMA.train()

    logger = additional['logger']


    # summary writer
    train_it = iter(data_loader)

    t_train = trange(0, args.iters, initial=0, total=args.iters)

    for i in t_train:
        try:
            imgs, y_org = next(train_it)
        except:
            train_it = iter(data_loader) # ...死循环
            print('TrainsetLen', len(data_loader))
            imgs, y_org = next(train_it) # images, class_idxs

        x_org = imgs
        x_ref_idx = torch.randperm(x_org.size(0)) # shuffle

        # x_org = x_org.cuda(args.gpu)

        # y_org = y_org.cuda(args.gpu)
        # x_ref_idx = x_ref_idx.cuda(args.gpu)

        x_org = x_org.to(torch.cuda.current_device())
        y_org = y_org.to(torch.cuda.current_device())
        x_ref_idx = x_ref_idx.to(torch.cuda.current_device())

        x_ref = x_org.clone()
        x_ref = x_ref[x_ref_idx]

        training_mode = 'GAN'

        ####################
        # BEGIN Train GANs #
        ####################
        with torch.no_grad():
            y_ref = y_org.clone()
            y_ref = y_ref[x_ref_idx]
            s_ref = C.moco(x_ref)
            c_src, skip1, skip2 = G.cnt_encoder(x_org)
            x_fake, _ = G.decode(c_src, s_ref, skip1, skip2)

        x_ref.requires_grad_()

        d_real_logit, _ = D(x_ref, y_ref)
        d_fake_logit, _ = D(x_fake.detach(), y_ref)

        d_adv_real = calc_adv_loss(d_real_logit, 'd_real')
        d_adv_fake = calc_adv_loss(d_fake_logit, 'd_fake')

        d_adv = d_adv_real + d_adv_fake

        d_gp = args.w_gp * compute_grad_gp(d_real_logit, x_ref, is_patch=False)

        d_loss = d_adv + d_gp

        d_opt.zero_grad()
        d_adv_real.backward(retain_graph=True)
        d_gp.backward()
        d_adv_fake.backward()
        if args.distributed:
            average_gradients(D)
        d_opt.step()

        # Train G
        s_src = C.moco(x_org)
        s_ref = C.moco(x_ref)

        c_src, skip1, skip2 = G.cnt_encoder(x_org)
        x_fake, offset_loss = G.decode(c_src, s_ref, skip1, skip2)
        x_rec, _ = G.decode(c_src, s_src, skip1, skip2)

        g_fake_logit, _ = D(x_fake, y_ref)
        g_rec_logit, _ = D(x_rec, y_org)

        g_adv_fake = calc_adv_loss(g_fake_logit, 'g')
        g_adv_rec = calc_adv_loss(g_rec_logit, 'g')

        g_adv = g_adv_fake + g_adv_rec

        g_imgrec = calc_recon_loss(x_rec, x_org) if not args.no_l1 else torch.zeros_like(g_adv)
        
        if phl:
            g_imgrec += calc_pseudo_hamming_loss(x_rec, x_org, thres=0) # -1 ~ 1
        
        if wdl:
            g_imgrec += calc_wdl(x_rec, x_org) * args.w_wdl # -1 ~ 1

        if args.pkl:
            g_imgrec += calc_pkl(x_rec, x_org) * args.w_pkl # -1 ~ 1

        # TODO Maybe add detach and binary clip?
        if quantize:
            x_fake = (x_fake*255).round()/255

        if detach:
            x_fake = x_fake.detach()

        # c_x_fake, _, _ = G.cnt_encoder(x_fake)
        c_x_fake, skip1_x_fake, skip2_x_fake = G.cnt_encoder(x_fake)
        g_conrec = calc_recon_loss(c_x_fake, c_src)

        g_loss = args.w_adv * g_adv + args.w_rec * g_imgrec +args.w_rec * g_conrec + args.w_off * offset_loss

        # abl
        if abl:
            g_img_abl = calc_abl(x_rec, x_org)
            # print(f"abl:{g_img_abl}  g_adv:{g_adv}  g_imgrec:{g_imgrec}  g_conrec:{g_conrec}  offset_loss:{offset_loss}")
            if g_img_abl is not None:
                g_loss += args.w_rec * g_img_abl

        if style_con:
            s_x_fake = C.moco(x_fake)
            g_stylerec = calc_recon_loss(s_ref, s_x_fake)
            g_loss = g_loss + args.w_rec * g_stylerec

        if reconstruction_losses:
            if not style_con:
                s_x_fake = C.moco(x_fake)
            c_ref, skip1_ref, skip2_ref = G.cnt_encoder(x_ref)

            # need recheck
            x_org_fake, _ = G.decode(c_x_fake, s_src.detach(), skip1_x_fake, skip2_x_fake)
            x_ref_fake, _ = G.decode(c_ref.detach(), s_x_fake, skip1_ref.detach(), skip2_ref.detach())

            # calc_recon_loss
            g_imgrec_org = calc_recon_loss(x_org_fake, x_org)
            g_imgrec_ref = calc_recon_loss(x_ref_fake, x_ref)
            g_loss = g_loss + args.w_rec * (g_imgrec_org + g_imgrec_ref)
        
        if hsic_loss:
            hsic_losses = add_indp_fact_loss(
                [s_src, c_src],
                [s_ref, c_ref],
                )
            g_loss = g_loss + args.w_hsic * (hsic_losses.mean())

        g_opt.zero_grad()
        c_opt.zero_grad()
        g_loss.backward()
        if args.distributed:
            average_gradients(G)
            average_gradients(C)
        c_opt.step()
        g_opt.step()

        ##################
        # END Train GANs #
        ##################


        if epoch >= args.ema_start:
            training_mode = training_mode + "_EMA"
            update_average(G_EMA, G)
        update_average(C_EMA, C)

        torch.cuda.synchronize()

        with torch.no_grad():
            if epoch >= args.separated:
                d_losses.update(d_loss.item(), x_org.size(0))
                d_advs.update(d_adv.item(), x_org.size(0))
                d_gps.update(d_gp.item(), x_org.size(0))

                g_losses.update(g_loss.item(), x_org.size(0))
                g_advs.update(g_adv.item(), x_org.size(0))
                g_imgrecs.update(g_imgrec.item(), x_org.size(0))
                g_rec.update(g_conrec.item(), x_org.size(0))

                moco_losses.update(offset_loss.item(), x_org.size(0))

            if (i + 1) % args.log_step == 0 and (args.gpu == 0 or args.gpu == '0') and logger is not None and args.local_rank == 0:
                summary_step = epoch * args.iters + i
                add_logs(args, logger, 'D/LOSS', d_losses.avg, summary_step)
                add_logs(args, logger, 'D/ADV', d_advs.avg, summary_step)
                add_logs(args, logger, 'D/GP', d_gps.avg, summary_step)

                add_logs(args, logger, 'G/LOSS', g_losses.avg, summary_step)
                add_logs(args, logger, 'G/ADV', g_advs.avg, summary_step)
                add_logs(args, logger, 'G/IMGREC', g_imgrecs.avg, summary_step)
                add_logs(args, logger, 'G/conrec', g_rec.avg, summary_step)

                add_logs(args, logger, 'C/OFFSET', moco_losses.avg, summary_step)

                print('Epoch: [{}/{}] [{}/{}] MODE[{}] Avg Loss: D[{d_losses.avg:.2f}] G[{g_losses.avg:.2f}] '.format(epoch + 1, args.epochs, i+1, args.iters,
                                                        training_mode, d_losses=d_losses, g_losses=g_losses))

    copy_norm_params(G_EMA, G)
    copy_norm_params(C_EMA, C)

