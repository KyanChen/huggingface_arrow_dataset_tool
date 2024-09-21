import glob
import cv2
import datasets
import numpy as np
import torch


def transform_data(examples):
	# 解码图片
	image = cv2.imdecode(np.frombuffer(examples['image'], np.uint8), cv2.IMREAD_COLOR)
	return {'image': image}


class MyDataset(torch.utils.data.Dataset):
	def __init__(self, file_folder):
		data_files = glob.glob(f'{file_folder}/*/*.arrow')
		self.dataset = datasets.load_dataset('arrow', data_files=data_files, split='train', keep_in_memory=True)

	def __len__(self):
		return len(self.dataset)

	def transform_data(self, examples):
		# 解码图片
		image = cv2.imdecode(np.frombuffer(examples['image'], np.uint8), cv2.IMREAD_COLOR)
		return {'image': image, 'label': examples['label']}

	def __getitem__(self, idx):
		examples = self.dataset[idx]
		if self.transform:
			examples = self.transform_data(examples)
		return examples


if __name__ == '__main__':
	# 根据dataset_generator.py的代码，生成arrow数据集，使用以下代码读取arrow数据集，并服务于Pytorch的模型训练
	arrow_folder = '/Users/kyanchen/Documents/vis_levirship/ab_yolov3_arrow'
	dataset = MyDataset(arrow_folder)
