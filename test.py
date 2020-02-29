import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import TestDataset, Resizer_test, Normalizer_test, collater_test
from src.efficientdet import EfficientDet
from config import get_args
from tqdm import tqdm
import pandas as pd


def test(opt):
    opt.resume = True
    test_set = TestDataset(opt.data_path, transform=transforms.Compose([Normalizer_test(), Resizer_test()]))

    opt.num_classes = test_set.num_classes
    opt.batch_size = opt.batch_size*4
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": collater_test,
                   "num_workers": 0}
    test_generator = DataLoader(test_set, **test_params)
    
    model = EfficientDet(opt)
    model.load_state_dict(torch.load(os.path.join(opt.pretrained_model, opt.network+'.pth')))
    model.cuda()
    model.set_is_training(False)
    model.eval()
    
    submission = {}
    submission['name'] = []
    submission['image_id'] = []
    submission['confidence'] = []
    submission['xmin'] = []
    submission['ymin'] = []
    submission['xmax'] = []
    submission['ymax'] = []

    progress_bar = tqdm(test_generator)
    progress_bar.set_description_str(' Testing')
    for i, data in enumerate(progress_bar):
        scale = data['scale']
        with torch.no_grad():
            output_list = model(data['img'].cuda().float())
        
        for j, output in enumerate(output_list):
            scores, labels, boxes = output

            if boxes.shape[0] == 0:
                continue
            
            boxes /= scale[j]
            imageName = test_set.getImageName(i*opt.batch_size+j)

            for box_id in range(boxes.shape[0]):
                pred_prob = float(scores[box_id])
                if pred_prob < opt.cls_threshold:
                    break
                pred_label = int(labels[box_id])
                xmin, ymin, xmax, ymax = boxes[box_id, :].cpu().numpy()

                pred_label = test_set.index2label(pred_label)

                submission['name'].append(pred_label)
                submission['image_id'].append(imageName[:-4]+'.xml')
                submission['confidence'].append(pred_prob)
                submission['xmin'].append(xmin)
                submission['ymin'].append(ymin)
                submission['xmax'].append(xmax)
                submission['ymax'].append(ymax)

    pd.DataFrame(submission).to_csv('submisson.csv', index=False)

if __name__ == "__main__":
    opt = get_args()
    test(opt)
                