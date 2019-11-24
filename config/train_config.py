import argparse
import os


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--data_root', default="D:/Vision_Images/CIFAR-10/batches", required=True, help='path to Dataset')
        self.parser.add_argument('--num_classes', default=10, required=True, help='number of classes in dataset', type=int)
        self.parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        self.parser.add_argument('--img_size', type=int, default=32, help='scale images to this size')
        self.parser.add_argument('--nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--checkpoints_dir', type=str, default='D:/DL_Models/Filip/', help='models are saved here')
        self.parser.add_argument('--drop_rate', default=0.3, type=float, help='drop rate for the network')
        self.parser.add_argument('--name', type=str, default='simplenet', choices=['simplenet', 'simplenet_residual', 'moe_vanila', 'moe_residual'], help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--learning_rate', '-lr', default=1e-3, type=float)
        self.parser.add_argument('--normalize', default=True, type=bool)
        self.parser.add_argument('--log_fq', default=10, type=int)
        self.parser.add_argument('--save_fq', default=200, type=int)
        self.parser.add_argument('--steps_per_epoch', default=1000, type=int)
        self.parser.add_argument('--epochs', default=20, type=int)
        self.parser.add_argument('--start_epoch', default=0, type=int)
        self.parser.add_argument('--mode', default='train', type=str)

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
