#  MIT License
#
#  Copyright (c) 2017 Ilya Kostrikov and (c) 2020 Google LLC
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--lr', type=float, default=3e-4, help='learning rate (default: 3e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer alpha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=True,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=10,
        help='number of ppo epochs (default: 10)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=64)
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=None,
        help='number of rollouts to train (default: None), overwrites num-env-steps')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='log/',
        help='directory to save agent logs (default: log/)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models_0/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--no-proper-time-limits',
        action='store_true',
        default=False,
        help='dont compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--warm-start',
        default='',
        help='policy pathname to warm start')
    parser.add_argument(
        '--warm-start-logstd',
        type=float,
        default=None,
        help='change warm start logstd to value if not None (default)')

    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-dyn',
        action='store_true',
        default=False,
        help='do dynamics imitation learning if gail on')
    parser.add_argument(
        '--gail-traj-path',
        default='',
        help='traj pathname for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch',
        type=int,
        default=5,
        help='gail discriminator epochs (default: 5)')
    parser.add_argument(
        '--gail-traj-num',
        type=int,
        default=20,
        help='gail batch size (default: 20)')
    parser.add_argument(
        '--gail-downsample-frequency',
        type=int,
        default=20,
        help='gail expert trajectory downsample frequency (default: 20)')
    parser.add_argument(
        '--gail-dis-hdim',
        type=int,
        default=100,
        help='gail D hidden dim (default: 100)')
    # parser.add_argument(
    #     '--gail-tar-length',
    #     type=float,
    #     default=100,
    #     help='gail demonstrations average episode length')
    parser.add_argument(
        '--no-alive-bonus',
        action='store_true',
        default=False,
        help='do not use alive bonus for gail dyn')
    parser.add_argument(
        '--use-split-pi',
        action='store_true',
        default=False,
        help='use split pi for gail dyn')
    parser.add_argument(
        '--num-feet',
        type=int,
        default=1,
        help='num of feet when use split pi hopper')

    parser.add_argument(
        '--dup-sym',
        action='store_true',
        default=False,
        help='use duplicate tuple symmetry')
    parser.add_argument(
        '--loss-sym',
        type=float,
        default=0.0,
        help='loss weight for mirror symmetry (default: 0.0, off)')

    args, extra_dict = parse_args_with_unknown(parser)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args, extra_dict


def parse_args_with_unknown(parser):
    args, unknown = parser.parse_known_args()  # this is an 'internal' method

    # which returns 'parsed', the same as what parse_args() would return
    # and 'unknown', the remainder of that
    # the difference to parse_args() is that it does not exit when it finds redundant arguments
    def try_numerical(string):
        # convert all extra arguments to numerical type (float) if possible
        # assume always float (pass bool as 0 or 1)
        # else, keep the argument as string type
        try:
            num = float(string)
            return num
        except ValueError:
            return string

    def pairwise(iterable):
        "s -> (s0, s1), (s2, s3), (s4, s5), ..."
        a = iter(iterable)
        return zip(a, a)

    for arg, value in pairwise(unknown):  # note: assume always --arg value (no --arg)
        assert arg.startswith(("-", "--"))
        parser.add_argument(arg, type=try_numerical)
    args_w_extra = parser.parse_args()
    args_dict = vars(args)
    args_w_extra_dict = vars(args_w_extra)
    # fix an issue: args_w_extra may have same key, diff value with args
    # for the cases of with-hyphen argument but pass in string with underscore
    args = args_w_extra
    extra_dict = {k: args_w_extra_dict[k] for k in set(args_w_extra_dict) - set(args_dict)}
    return args, extra_dict
