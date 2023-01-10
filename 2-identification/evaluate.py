import torch
import torch.nn as nn
from utils.dataset import ChimpDataset
from models.resnet import resnet18
import argparse
import os
from utils.utils import *
import time
from torchvision import transforms
from tqdm import tqdm
from utils.loss import nt_xent, sup_nt_xent
from torch.utils.tensorboard import SummaryWriter
from utils.loss import FocalLoss

parser = argparse.ArgumentParser(description='Chimp identification evaluation')
parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--pretrained_weights',type=str,
                    default='./pretrained_weights/model_new.pt',help='checkpoint to load')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def validate(model, loaders):
	model.eval()
	criterion = nn.CrossEntropyLoss()
	criterion = criterion.cuda()

	top1 = AverageMeter()
	losses = AverageMeter()

	acc = []

	for loader in loaders:
		losses.reset()
		top1.reset()
		for images, labels in loader:
			images = images.cuda()
			labels = labels.cuda()

			with torch.no_grad():
				outputs = model(images)
				loss = criterion(outputs,labels)

			prec1 = accuracy(outputs.data, labels)[0]
			top1.update(prec1.item(), images.size(0))
			losses.update(float(loss.detach().cpu()))
		acc.append(top1.avg)

	return acc

def main():
	transforms_test = transforms.Compose([transforms.ToTensor(),
								transforms.Resize((224,224)),
								transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
 
	test_dataset = ChimpDataset(split='val',transform=transforms_test)

	test_loader = torch.utils.data.DataLoader(test_dataset,
			num_workers=16,
			batch_size=32,
			shuffle=False)

	model = resnet18(pretrained=False)
	model.fc = nn.Linear(512, 17)
 
	model.load_state_dict(torch.load(args.pretrained_weights), strict=False)
  
	model.cuda().eval()

	acc = validate(model, [test_loader])
	
	print(f'Top-1 test accuracy: {acc[0]:.2f}')
	
if __name__ == '__main__':
	main()
