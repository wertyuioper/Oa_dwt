import os
import math
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset自定义数据集 import ViTDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
from utils读取数据集 import read_split_data, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 创建 'weights' 目录（如果不存在）
    os.makedirs("./weights", exist_ok=True)

    # TensorBoard 记录器
    tb_writer = SummaryWriter(log_dir="./runs4/ViT_experiment")

    # 准备数据集路径
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # 图像预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    # 实例化训练和验证数据集
    train_dataset = ViTDataSet(images_path=train_images_path, images_class=train_images_label,
                               transform=data_transform["train"])
    val_dataset = ViTDataSet(images_path=val_images_path, images_class=val_images_label,
                             transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 设置工作进程数
    print(f'每个进程使用 {nw} 个 dataloader 工作线程')

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                               num_workers=nw, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                             num_workers=nw, collate_fn=val_dataset.collate_fn)

    # 加载模型
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    # 加载预训练权重
    if args.weights != "":
        assert os.path.exists(args.weights), f"权重文件: '{args.weights}' 不存在。"
        weights_dict = torch.load(args.weights, map_location=device)

        # 删除不必要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits else ['pre_logits.fc.weight', 'pre_logits.fc.bias',
                                                                          'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]

        print(model.load_state_dict(weights_dict, strict=False))

    # 冻结层（如果需要）
    if args.freeze_layers:
        for name, param in model.named_parameters():
            if "head" not in name and "pre_logits" not in name:
                param.requires_grad_(False)
            else:
                print(f"训练 {name}")

    # 优化器和学习率调度器
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=0.9, weight_decay=5E-5)

    # 余弦退火学习率调度
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 开始训练和验证
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                                device=device, epoch=epoch)

        # 调整学习率
        scheduler.step()

        # 验证
        val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch)

        # 记录指标到 TensorBoard
        tb_writer.add_scalar("Loss/Train", train_loss, epoch)
        tb_writer.add_scalar("Accuracy/Train", train_acc, epoch)
        tb_writer.add_scalar("Loss/Validation", val_loss, epoch)
        tb_writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        tb_writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

        # 每个 epoch 保存模型
        #torch.save(model.state_dict(), f"./weights/model_epoch_{epoch}.pth")

    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)  # 输出类数量
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集路径
    parser.add_argument('--data-path', type=str, default='data_set3/flower_photos')

    # 模型名称（可选）
    parser.add_argument('--model-name', default='', help='要创建的模型名称')

    # 预训练权重路径
    parser.add_argument('--weights', type=str, default='',
                        help='初始权重路径')#jx_vit_base_patch16_224_in21k-e5005f0a.pth

    # 是否冻结层
    parser.add_argument('--freeze-layers', type=bool, default=True)

    # 设备配置
    parser.add_argument('--device', default='cuda:0', help='设备 id (例如 0 或 0,1 或 cpu)')

    opt = parser.parse_args()
    main(opt)







