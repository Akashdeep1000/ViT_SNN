!pip install einops
!pip install snntorch

import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity

from einops import rearrange
import torchvision
import torchvision.models as models

import time

import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate

from tqdm import tqdm

import matplotlib.pyplot as plt

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('GPU: ', torch.cuda.get_device_name(0))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        '''
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        '''
        self.ff_fc1 = nn.Linear(dim, hidden_dim)
        self.ff_lif1 = snn.Leaky(beta = 0.95, learn_beta=True, learn_threshold=True)
        self.ff_fc2 = nn.Linear(hidden_dim, dim)
        

    def forward(self, x):
        
        mem1 = self.ff_lif1.init_leaky()

        x = self.ff_fc1(x)
        x, mem1 = self.ff_lif1(x, mem1)
        
        x = self.ff_fc2(x)
        return x
        
        #return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()
        
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )
        
    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

def train(model, dataloader, criterion, optimizer, scheduler, resnet_features=None):
    running_loss = 0.0
    running_accuracy = 0.0

    for data, target in tqdm(dataloader):
        data = data.to(device)
        target = target.to(device)
        data = spikegen.rate_conv(data)
        model.train()

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        acc = (output.argmax(dim=1) == target).float().mean()
        running_accuracy += acc / len(dataloader)
        running_loss += loss.item() / len(dataloader)

    return running_loss, running_accuracy

def evaluation(model, dataloader, criterion, resnet_features=None):
    with torch.no_grad():
        test_accuracy = 0.0
        test_loss = 0.0
        for data, target in tqdm(dataloader):
            data = data.to(device)
            target = target.to(device)
            data = spikegen.rate_conv(data)
            model.eval()

            output = model(data)
            loss = criterion(output, target)

            acc = (output.argmax(dim=1) == target).float().mean()
            test_accuracy += acc / len(dataloader)
            test_loss += loss.item() / len(dataloader)

    return test_loss, test_accuracy

DOWNLOAD_PATH = '/data/mnist'
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 1000
torch.manual_seed(42)

transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0,), (1,))])

train_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=True, download=True, transform=transform_mnist)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

test_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True, transform=transform_mnist)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)

N_EPOCHS = 25
lr=0.003
PATCH_SIZE = 7
DIM = 64
DEPTH = 8
HEADS = 16
MLP_DIM = 128

start_time = time.time()
model = ViT(image_size=28, patch_size=PATCH_SIZE, num_classes=10, channels=1, dim=DIM, depth=DEPTH, heads=HEADS, mlp_dim=MLP_DIM)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss().to(device)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = lr, steps_per_epoch=len(train_loader), epochs=N_EPOCHS)

train_loss_history, test_loss_history = [], []

train_accs = []
test_accs = []

for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)

    running_loss, running_accuracy = train(model, train_loader, criterion, optimizer, scheduler)
    print(f"Epoch : {epoch+1} - acc: {running_accuracy:.4f} - loss : {running_loss:.4f}\n")
    train_accs.append(running_accuracy)

    test_loss, test_accuracy = evaluation(model, test_loader, criterion)
    print(f"test acc: {test_accuracy:.4f} - test loss : {test_loss:.4f}\n")
    test_accs.append(test_accuracy)

print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

train_accs = [acc.cpu().item() for acc in train_accs]
test_accs = [acc.cpu().item() for acc in test_accs]

plt.style.use('seaborn')
plt.plot(range(1, N_EPOCHS+1), train_accs, label='Train Accuracy')
plt.plot(range(1, N_EPOCHS+1), test_accs, label='Test Accuracy')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.title("")
plt.title("HYBRID MNIST\nTrain vs Test Accuracy\nSpiking Only in Feed Forward\n" + "BATCH SIZE = " + str(BATCH_SIZE_TRAIN) + ", LR = " + str(lr) + ", PATCH SIZE = " + str(PATCH_SIZE) + ", DIM = " + str(DIM) +
          ", DEPTH = " + str(DEPTH) + ", HEADS = " + str(HEADS) + ", MLP DIM = " + str(MLP_DIM))
plt.legend(loc='lower right')
plt.show()
torch.save(model.state_dict(), 'MNISTHybridNoTime.pth')