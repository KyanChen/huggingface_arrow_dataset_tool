import glob
import mmengine
from datasets import Dataset
import os
import cv2


def convert_data2dict(examples):
	img_file, label = examples
	img = cv2.imread(img_file, cv2.IMREAD_COLOR)
	# convert img to bytes
	img_bytes = cv2.imencode('.png', img)[1].tobytes()
	data_dict = {
		"image": img_bytes,
		"label": label,
		'file_name': os.path.basename(img_file)
	}
	return data_dict


if __name__ == '__main__':
	# 假如您的任务是图像分类，数据是百万级别的图像数据，您可以使用以下代码生成arrow数据集

	img_folder = '/Users/kyanchen/Documents/vis_levirship/ab_yolov3'
	arrow_folder = '/Users/kyanchen/Documents/vis_levirship/ab_yolov3_arrow'
	max_samples_per_shard = 100000  # 每个shard最多存储的样本数，最好10W左右
	n_proc = 8  # 多进程处理数据，8，16都行

	mmengine.mkdir_or_exist(arrow_folder)
	img_files = glob.glob(f'{img_folder}/*.png')
	labels = [os.path.basename(x).split('_')[0] for x in img_files]
	data_list = list(zip(img_files, labels))

	# 生成arrow数据集
	for i_start in range(0, len(data_list), max_samples_per_shard):
		print(f"Processing {i_start} to {i_start + max_samples_per_shard}, total {len(data_list)}")
		i_end = min(i_start + max_samples_per_shard, len(data_list))
		data_list_shard = data_list[i_start: i_end]

		results = mmengine.track_parallel_progress(convert_data2dict, data_list_shard, nproc=n_proc)
		# 保存arrow数据集
		dataset = Dataset.from_list(results)
		dataset.save_to_disk(f'{arrow_folder}/data_{i_start}_{i_end}', num_proc=1, max_shard_size='4GB')
		print(f"Save to {arrow_folder}/data_{i_start}_{i_end}")


