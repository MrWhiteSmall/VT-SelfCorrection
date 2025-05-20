from pathlib import Path
import sys,os,random,torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
# 将项目地址放入 系统变量 中
sys.path.insert(0, '/data/lsj/OOTDiffusion/SCNet')
sys.path.insert(0, str(PROJECT_ROOT))
# print(sys.path)

prefix = '/data/lsj/OOTDiffusion/run'
cloth_path = '/data/lsj/OOTDiffusion/run/examples/garment/00055_00.jpg'
model_path = '/data/lsj/OOTDiffusion/run/examples/model/01008_00.jpg'

def join_path(*args):
    path = ''
    for a in args:
        path=os.path.join(path,a)
    return path


from utils_ootd import get_mask_location
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD

# from SCNet.main_test_ltcc import main_ltcc


import argparse

def get_args():
    parser = argparse.ArgumentParser(description='run ootd')
    parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
    # parser.add_argument('--model_path', type=str, default="", required=True)
    # parser.add_argument('--cloth_path', type=str, default="", required=True)
    parser.add_argument('--model_type', type=str, default="hd", required=False)
    parser.add_argument('--category', '-c', type=int, default=0, required=False)
    parser.add_argument('--scale', type=float, default=2.0, required=False)
    parser.add_argument('--step', type=int, default=20, required=False)
    parser.add_argument('--sample', type=int, default=4, required=False)
    parser.add_argument('--seed', type=int, default=-1, required=False)
    args = parser.parse_args()
    return args

def get_pose_parse_model(args):
    openpose_model = OpenPose(args.gpu_id)
    parsing_model = Parsing(args.gpu_id)
    return openpose_model,parsing_model

def get_vton_model(gpu_id):
    model = OOTDiffusionHD(gpu_id)
    return model

def get_vton_res(cloth_path,model_path,save=False,no_grad = False):
    prefix = '/data/lsj/OOTDiffusion/run'
    # 只要将cloth path处理的步骤放到外面即可
    args = get_args()
    # image_scale = args.scale
    # n_steps = args.step
    # n_samples = args.sample
    seed = args.seed
    
    # target_shape = (768, 1024)
    # target_shape = (768//2, 1024//2)
    # target_shape = (768//4, 1024//4)
    target_shape = (192 , 384)

    with torch.no_grad():
        openpose_model,parsing_model = get_pose_parse_model(args)
        # model = get_vton_model(args)

        cloth_img = Image.open(cloth_path).resize(target_shape)
        model_img = Image.open(model_path).resize(target_shape)
        keypoints = openpose_model(model_img.resize((768//2, 1024//2)))
        model_parse, _ = parsing_model(model_img.resize((768//2, 1024//2)))

        mask, mask_gray = get_mask_location('hd', 'upper_body', model_parse, keypoints)
        mask = mask.resize(target_shape, Image.NEAREST)
        mask_gray = mask_gray.resize(target_shape, Image.NEAREST)
        
        masked_vton_img = Image.composite(mask_gray, model_img, mask)
        print('保存图片 start',cloth_path,model_path)
        if save:
            masked_vton_img.save( join_path(prefix,'images_output/mask.jpg'))

    # images = model(
    #     model_type='hd',
    #     category='upperbody',
    #     image_garm=cloth_img,
    #     image_vton=masked_vton_img,
    #     mask=mask,
    #     image_ori=model_img,
    #     num_samples=4,
    #     num_steps=20,
    #     image_scale=2.0,
    #     seed=seed,
    #     no_grad = no_grad, # False : 需要grad True : 不需要grad
    # )
    

    print('保存图片 end',cloth_path,model_path)
    # if save:
    #     images[0].save( join_path(prefix, 'images_output/out_' + 'hd'  + '.png' )   )

    return cloth_img,masked_vton_img,mask,model_img

def get_vton_res_by_img(cloth_img,model_img,save=False):
    prefix = ''
    # 只要将cloth path处理的步骤放到外面即可
    args = get_args()
    image_scale = args.scale
    n_steps = args.step
    n_samples = args.sample
    seed = args.seed

    openpose_model,parsing_model = get_pose_parse_model(args)
    model = get_vton_model(args)

    # cloth_img = Image.open(cloth_path).resize((768, 1024))
    # model_img = Image.open(model_path).resize((768, 1024))
    keypoints = openpose_model(model_img.resize((384, 512)))
    model_parse, _ = parsing_model(model_img.resize((384, 512)))

    mask, mask_gray = get_mask_location('hd', 'upperbody', model_parse, keypoints)
    mask = mask.resize((768, 1024), Image.NEAREST)
    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
    
    masked_vton_img = Image.composite(mask_gray, model_img, mask)
    print('保存图片 start')
    if save:
        masked_vton_img.save('./images_output/mask.jpg')
        masked_vton_img.save( join_path(prefix,'images_output/mask.jpg'))

    images = model(
        model_type='hd',
        category='upperbody',
        image_garm=cloth_img,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=model_img,
        num_samples=4,
        num_steps=20,
        image_scale=2.0,
        seed=seed,
    )

    print('保存图片 end')
    if save:
        images[0].save( join_path(prefix, 'images_output/out_' + 'hd'  + '.png' )   )

    return images[0]


# cloth_dir = '/public/backup/test04/lsj/datasets/VITON-HD/train/cloth'
# cloth_names = os.listdir(cloth_dir)
# def get_random_cloth_path():
#     random.shuffle(cloth_names)
#     cloth_path = os.path.join(cloth_dir,cloth_names[0])
#     return cloth_path

def try_main_vcc():
    from SCNet.main_test_vcc import main_vcc
    
    top1,top5,top10,top20,mAP = main_vcc(get_vton_model,get_vton_by_model=get_vton_res)
    # 传入vton-model，cloth-path，以及get-vton-res的方法
    # top1,top5,top10,top20,mAP = 0,0,0,0,0
    # count = 1
    # while top20 == 0 or count < 100:
    #     # cloth_path = get_random_cloth_path()
    #     top1,top5,top10,top20,mAP = main_vcc(cloth_path=cloth_path,get_vton_by_model=get_vton_res)
    #     count += 1

    # if count == 100:
    #     print('不太行，100次也没找到')
    # else:
    #     print('这件衣服比较好detect')
    #     print(cloth_path)
    #     print(top1,top5,top10,top20,mAP)

if __name__ == '__main__':
   
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try_main_vcc()
    
    # main_ltcc()

    # img,optimizer = get_vton_res(cloth_path=cloth_path,model_path=model_path,save=True,no_grad = False)
    # # 定义损失函数
    # criterion = nn.CrossEntropyLoss()
    
    # # 计算损失
    # loss = criterion(output, targets)

    # # 反向传播并更新小模型的参数
    # optimizer.zero_grad()  # 清空梯度
    # loss.backward()  # 计算梯度
    # optimizer.step()  # 更新小模型的参数

    # 打印当前显存分配情况
    print(f"当前显存占用: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")

    # 打印运行过程中最大的显存分配情况
    print(f"最大显存占用: {torch.cuda.max_memory_allocated(device) / (1024 ** 2):.2f} MB")


# 使用 torch.cuda.memory_allocated 和 torch.cuda.max_memory_allocated
# torch.cuda.memory_allocated(): 返回当前在 GPU 上分配的显存量。
# torch.cuda.max_memory_allocated(): 返回运行过程中在 GPU 上分配的最大显存量。