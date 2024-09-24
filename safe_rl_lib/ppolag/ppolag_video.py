import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

import numpy as np
import torch
from torch.optim import Adam
import gym
import gymnasium
import time
 
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def atari_env_fn(atari_name, version='5'):
    env_name = 'ALE/' + atari_name + '-v' + version
    return gymnasium.make(env_name, obs_type="ram", render_mode='rgb_array')


def replay(env_fn, model_path=None, video_name=None, max_epoch=1, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not model_path:
        print("please specify a model path")
        raise NotImplementedError
    if not video_name:
        print("please specify a video name")
        raise NotImplementedError    
    
    # Instantiate environment
    env = env_fn()
    
    # reset environment
    o = env.reset(seed=seed)
    d = False
    ep_ret = 0
    time_step = 0
    epoch = 0
    
    video_array = []
    
    # load the model 
    ac = torch.load(model_path)
    
    # evaluate the model 
    while True:
        time_step += 1
        if d:
            epoch += 1
            print('Episode Return: %.3f'%(ep_ret))
            if epoch == max_epoch:
                env.close()
                break
            ep_ret = 0
            o = env.reset(seed=seed)
        
        if isinstance(o, tuple):
            o = o[0]
        a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
        

        next_o, r, d1, d2, _ = env.step(a)
        d = d1 or d2
        
        # Update obs (critical!)
        o = next_o

        img_array = env.render()
        video_array.append(img_array)

        ep_ret += r

    # save video 
    fps = 60
    dsize = (1080,1920)
    out_path = '../video'
    existence = os.path.exists(out_path)
    if not existence:
        os.makedirs(out_path)
    video_writer = cv2.VideoWriter(os.path.join(out_path,f'{video_name}.mp4'),
                                cv2.VideoWriter_fourcc(*'FMP4'), fps, dsize)

    for frame in video_array:
        resized = cv2.resize(frame, dsize=dsize)
        video_writer.write(resized)

    video_writer.release()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()    
    parser.add_argument('--task', type=str, default='Goal_Point')
    parser.add_argument('--max_epoch', type=int, default=1)  # the maximum number of epochs
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--video_name', type=str, default=None)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    replay(lambda : atari_env_fn(args.task), model_path=args.model_path, video_name=args.video_name, max_epoch=args.max_epoch, seed=args.seed)