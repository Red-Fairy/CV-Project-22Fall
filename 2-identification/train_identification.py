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

parser = argparse.ArgumentParser(description='Chimp identification training')
parser.add_argument('--experiment', type=str,
					help='location for saving trained models')
parser.add_argument('--epochs', default=50, type=int,
					help='number of total epochs to run')
parser.add_argument('--lr', default=1e-4, type=float, help='optimizer lr')
parser.add_argument('--bsz', default=32, type=int, help='batch size')
parser.add_argument('--gpu_id', default='4', type=str)
parser.add_argument('--resnet_checkpoint',type=str,
                    default='/home/luord/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth',
                    help='checkpoint to load')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
save_dir = os.path.join('experiments', args.experiment)
if os.path.exists(save_dir) is not True:
	os.system("mkdir -p {}".format(save_dir))
log = logger(path=save_dir)
log.info(str(args))
writer = SummaryWriter(log_dir=save_dir)


def train_contrastive(train_loader, model, optimizer, scheduler, epoch):
	model.train()
	losses, losses_task, losses_con = AverageMeter(), AverageMeter(), AverageMeter()
	losses.reset(), losses_task.reset(), losses_con.reset()
	criterion = torch.nn.CrossEntropyLoss()
	st = time.time()

	for i, (images, images_cls, labels) in enumerate(tqdm(train_loader)):
		images = images.cuda()
		images_cls = images_cls.cuda()
		labels = labels.cuda()
		d = images.shape
		images = images.view(d[0]*2, d[2], d[3], d[4])

		_, outputs_feature = model(images, proj=True)
		outputs = model(images_cls)

		loss = 0.

		loss_task = criterion(outputs, labels)
		losses_task.update(float(loss_task.detach().cpu()), images.shape[0])
		loss += loss_task

		# loss_con = nt_xent(outputs_feature)
		loss_con = sup_nt_xent(outputs_feature, labels.expand(2, -1).T.reshape(-1))
		losses_con.update(float(loss_con.detach().cpu()), images.shape[0])
		loss += loss_con

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.update(float(loss.detach().cpu()), images.shape[0])

	train_time = time.time() - st
	log.info(f'Epoch: {epoch}\t Loss: {losses.avg:.4f}\t Loss_task: {losses_task.avg:.4f}\t Loss_con: {losses_con.avg:.4f}\t Time: {train_time:.2f}')
	writer.add_scalar('Loss/train', losses.avg, epoch)
	writer.add_scalar('Loss/train_task', losses_task.avg, epoch)
	writer.add_scalar('Loss/train_con', losses_con.avg, epoch)
	scheduler.step()

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
		log.info(f"Accuracy: {top1.avg:.2f}\t Loss: {losses.avg:.4f}")

	return acc

def main():
	transforms_train_contrastive = transforms.Compose([
								transforms.RandomResizedCrop(224, (0.8, 1.0)),
								transforms.RandomRotation(15),
								transforms.RandomHorizontalFlip(p=0.5),
								transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
								transforms.ToTensor(),
								transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
	transforms_train = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
								transforms.Resize((224,224)),
								transforms.ToTensor(),
								transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
	transforms_test = transforms.Compose([transforms.ToTensor(),
								transforms.Resize((224,224)),
								transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

	train_con_dataset = ChimpDataset(split='train',transform=transforms_train, contrastive=True, transform_contrastive=transforms_train_contrastive)
	# train_dataset = ChimpDataset(split='train',transform=transforms_train)
	val_dataset = ChimpDataset(split='train',transform=transforms_test)
	test_dataset = ChimpDataset(split='val',transform=transforms_test)

	train_con_loader = torch.utils.data.DataLoader(train_con_dataset,
			num_workers=16,
			batch_size=args.bsz,
			shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_dataset,
			num_workers=16,
			batch_size=args.bsz,
			shuffle=False)
	test_loader = torch.utils.data.DataLoader(test_dataset,
			num_workers=16,
			batch_size=args.bsz,
			shuffle=False)

	model = resnet18(pretrained=False, projection=True)
	# load the pretrained resnet
	model.load_state_dict(torch.load(args.resnet_checkpoint), strict=False)
	model.fc = nn.Linear(512, 17)
	model.cuda()

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)

	test_best_acc = 0.

	for epoch in range(1, args.epochs+1):
		log.info("current lr is {}".format(
			optimizer.state_dict()['param_groups'][0]['lr']))

		train_contrastive(train_con_loader,model,optimizer,scheduler,epoch)

		acc = validate(model, [val_loader,test_loader])
		
		if test_best_acc < acc[1]:
			test_best_acc = acc[1]
			save_checkpoint({
				'epoch': epoch,
				'state_dict': model.state_dict(),
				'optim': optimizer.state_dict(),
				'test_best_acc': test_best_acc,
			}, filename=os.path.join(save_dir, 'model_best.pt'))

		save_checkpoint({
			'epoch': epoch,
			'state_dict': model.state_dict(),
			'optim': optimizer.state_dict(),
		}, filename=os.path.join(save_dir, 'model.pt'))

	log.info(f'best acc: {acc[0]:.2f}')

if __name__ == '__main__':
	main()
