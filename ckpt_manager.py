"""
Authors : inzapp

Github url : https://github.com/inzapp/ckpt-manager

Copyright (c) 2023 Inzapp

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import shutil as sh

from glob import glob


class CheckpointManager:
    def __init__(self):
        self.model_name = None
        self.checkpoint_path = None

    def set_model_name(self, model_name):
        self.model_name = model_name

    def parse_pretrained_iteration_count(self, pretrained_model_path):
        iteration_count = 0
        sp = f'{os.path.basename(pretrained_model_path)[:-3]}'.split('_')
        for i in range(len(sp)):
            if sp[i] == 'iter' and i > 0:
                try:
                    iteration_count = int(sp[i-1])
                except:
                    pass
                break
        return iteration_count

    def make_checkpoint_dir(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def init_checkpoint_dir(self):
        inc = 0
        while True:
            if inc == 0:
                new_checkpoint_path = f'checkpoint/{self.model_name}'
            else:
                new_checkpoint_path = f'checkpoint/{self.model_name}_{inc}'
            if os.path.exists(new_checkpoint_path) and os.path.isdir(new_checkpoint_path):
                inc += 1
            else:
                break
        self.checkpoint_path = new_checkpoint_path
        self.make_checkpoint_dir()
        print(f'checkpoint path : {self.checkpoint_path}')

    def remove_last_model(self):
        for last_model_path in glob(f'{self.checkpoint_path}/model_*_iter.h5'):
            os.remove(last_model_path)

    def save_last_model(self, model, iteration_count):
        self.make_checkpoint_dir()
        save_path = f'{self.checkpoint_path}/model_{iteration_count}_iter.h5'
        model.save(save_path, include_optimizer=False)
        backup_path = f'{save_path}.bak'
        sh.move(save_path, backup_path)
        self.remove_last_model()
        sh.move(backup_path, save_path)
        return save_path

