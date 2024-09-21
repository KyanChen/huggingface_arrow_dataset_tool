import glob
import cv2
import datasets
import numpy as np


def transform_data(examples):
	# 解码图片
	image = cv2.imdecode(np.frombuffer(examples['image'], np.uint8), cv2.IMREAD_COLOR)
	return {'image': image}


if __name__ == '__main__':
	# 根据dataset_generator.py的代码，生成arrow数据集，使用以下代码读取arrow数据集，并服务于Huggingface的模型训练
	arrow_folder = '/Users/kyanchen/Documents/vis_levirship/ab_yolov3_arrow'
	num_proc = 8  # 多进程处理数据

	data_files = glob.glob(f'{arrow_folder}/*/*.arrow')
	# 读取arrow数据集
	dataset = datasets.load_dataset('arrow', data_files=data_files, split='train', keep_in_memory=True)
	print(dataset)
	# 预处理数据集
	dataset = dataset.map(transform_data, num_proc=num_proc, keep_in_memory=True)
	print(dataset)