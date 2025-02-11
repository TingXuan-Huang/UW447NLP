#!/usr/bin/env python
import os
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import train
import inference
import model

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        #NOT YET IMPLEMENTED
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        
        print('Instantiating model')
        model = model.CharGPT()
        
        print('Loading training data')
        train_data = train.load_training_data()
        
        print('Training')
        model.run_train(train_data, args.work_dir)
        
        print('Saving model')
        model.save(args.work_dir)
    
    elif args.mode == 'test':
        print('Loading model')
        model_path = os.path.join(args.work_dir, 'model.pth')
        my_model = inference.load_model(model_path, vocab_size = model.vocab_size)
        
        print('Loading test data from {}'.format(args.test_data))
        test_data = inference.load_test_data(args.test_data)

        print(test_data)
        print('Making predictions')
        pred, prob = inference.preict_test_data(my_model, test_data)
        
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        inference.write_pred(pred, args.test_output)
    
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
