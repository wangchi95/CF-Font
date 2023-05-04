from numpy import chararray
import click
import os

@click.command()
@click.argument('f1', type=click.File('r'))
@click.argument('f2', type=click.File('r'))

def check(f1, f2):
    t1 = set(f1.readlines()[0])
    t2 = set(f2.readlines()[0])
    print(len(t1), len(t2))
    print('Overlap:', len(t1 & t2))

if __name__ == "__main__":
    check()