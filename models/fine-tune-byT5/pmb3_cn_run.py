#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DRG_parsing 
@File ：run.py
@Author ：xiao zhang
@Date ：2022/11/14 12:27
'''

import argparse
import os
import sys
sys.path.append(".")

from pmb3_model import get_dataloader, Generator
#from pmb3_model import Dataset

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lang", required=False, type=str, default="en",
                        help="language in [en, nl, de ,it]")
    parser.add_argument("-pt", "--pretrain", required=False, type=str,
                        default=os.path.join(path, "data/pmb-3.0.0/train/train_cn_x2.txt"),
                        help="DRS file")
    parser.add_argument("-pt2", "--pretrain2", required=False, type=str,
                        default=os.path.join(path, "data/pmb-3.0.0/train/train_cn_x2.txt.raw"),
                        help="text input file")
    parser.add_argument("-t", "--train", required=False, type=str,
                        default=os.path.join(path, "data/pmb-3.0.0/train/train.txt"),
                        help="text input file")
    parser.add_argument("-t2", "--train2", required=False, type=str,
                        default=os.path.join(path, "data/pmb-3.0.0/train/train.txt.raw"),
                        help="drs input file")
    parser.add_argument("-dti", "--dev", required=False, type=str,
                        default=os.path.join(path, "data/pmb-3.0.0/dev/dev.txt"),
                        help="dev text input file")
    parser.add_argument("-dti2", "--dev2", required=False, type=str,
                        default=os.path.join(path, "data/pmb-3.0.0/dev/dev.txt.raw"),
                        help="dev drs input file")
    parser.add_argument("-tti", "--test", required=False, type=str,
                        default=os.path.join(path, "data/pmb-3.0.0/test/test.txt"),
                        help="test text input file")
    parser.add_argument("-tti2", "--test2", required=False, type=str,
                        default=os.path.join(path, "data/pmb-3.0.0/test/test.txt.raw"),
                        help="test text input file")
    parser.add_argument("-s", "--save1", required=False, type=str,
                        default=os.path.join(path, "src/model/byT5/pmb3_all_results/pmb3_cn_results/byT5_pmb3_gold.txt"),
                        help="path to save the result")
    parser.add_argument("-s2", "--save2", required=False, type=str,
                        default=os.path.join(path, "src/model/byT5/pmb3_all_results/pmb3_cn_results/byT5_en_long.txt"),
                        help="path to save the second result")
    parser.add_argument("-tl", "--test_long", required=False, type=str,
                        default="false",
                        help="path to save the second result")
    parser.add_argument("-m", "--mode", required=False, type=str,
                        default="train",
                        help="train or test")
    args = parser.parse_args()
    return args


def main():
    args = create_arg_parser()

    # train process
    lang = args.lang
    train_dataloader1 = get_dataloader(drs_file_path=args.pretrain, text_file_path=args.pretrain2)
    train_dataloader2 = get_dataloader(drs_file_path=args.train, text_file_path=args.train2)
    test_dataloader = get_dataloader(drs_file_path=args.test, text_file_path=args.test2)
    dev_dataloader = get_dataloader(drs_file_path=args.dev, text_file_path=args.dev2)

    save_path1 = args.save1
   # save_path2 = args.save2

    # mode
    mode = args.mode

    print(lang)

    test_long = args.test_long
    if mode == "train":
        if test_long == "true":
           # test_dataloader2 = get_dataloader(args.test2)
            bart_classifier = Generator(lang)
            bart_classifier.train(train_dataloader1, test_dataloader, lr=0.0001, epoch_number=10)
            bart_classifier.train(train_dataloader2, test_dataloader, lr=0.0001, epoch_number=5)
            bart_classifier.evaluate(dev_dataloader, save_path1+".dev")
            bart_classifier.evaluate(test_dataloader, save_path1)
           # bart_classifier.evaluate(test_dataloader2, save_path2)
        else:
            bart_classifier = Generator(lang)
            bart_classifier.train(train_dataloader1, test_dataloader, lr=0.0001, epoch_number=10)
            bart_classifier.train(train_dataloader2, test_dataloader, lr=0.0001, epoch_number=5)
            bart_classifier.evaluate(dev_dataloader, save_path1+".dev")
            bart_classifier.evaluate(test_dataloader, save_path1)

        bart_classifier.model.save_pretrained(os.path.join(path, f"models/pmb3_cn_byT5_seq2seq/{lang}"))
    else:
        bert_classifier = Generator()


if __name__ == '__main__':
    main()
