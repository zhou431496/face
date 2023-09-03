import collections

import torch
import torch.nn as nn
import clip
import torchvision.transforms as transforms
import torch.nn.functional as F


class ResNetEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.visual_model = clip_model.visual
        layers=list(self.visual_model.children())
        init_layers=torch.nn.Sequential(*layers)[:8]
        self.layer1=layers[8]#[128 112 112]
        self.layer2=layers[9]#[256 56 56]
        self.layer3=layers[10]#[512 28 28]
       
       
      

    def transf_to_CLIP_input(self,inputs):
        device = inputs.device
        if len(inputs.size()) != 4:
            raise ValueError('Expect the (B, C, X, Y) tensor.')
        else:
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])\
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            var = torch.tensor([0.26862954, 0.26130258, 0.27577711])\
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            inputs = F.interpolate(inputs*0.5+0.5, size=(224, 224))
            inputs = ((inputs+1)*0.5-mean)/var
            return inputs

    def forward(self, x: torch.Tensor):
        def stem(m,x):
            for conv ,bn in [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)]:
               x=m.relu(bn(conv(x)))
            x=m.avgpool(x)
            return x
        x = self.transf_to_CLIP_input(x)
        x = x.type(self.visual_model.conv1.weight.dtype)
        x= stem(self.visual_model, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        return x2,x3


class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        model = clip_model.visual
        # print(model)
        self.define_module(model)
        #for param in self.parameters():
         #   param.requires_grad = False

    def define_module(self, model):
        self.conv1 = model.conv1
        self.class_embedding = model.class_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_pre = model.ln_pre
        self.transformer = model.transformer
        self.ln_post = model.ln_post
        self.proj = model.proj

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def transf_to_CLIP_input(self,inputs):
        device = inputs.device
        if len(inputs.size()) != 4:
            raise ValueError('Expect the (B, C, X, Y) tensor.')
        else:
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])\
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            var = torch.tensor([0.26862954, 0.26130258, 0.27577711])\
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            inputs = F.interpolate(inputs*0.5+0.5, size=(224, 224))
            inputs = ((inputs+1)*0.5-mean)/var
            return inputs

    def forward(self, img: torch.Tensor):
        x = self.transf_to_CLIP_input(img)
        x = x.type(self.dtype)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        grid =  x.size(-1)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        # NLD -> LND
        x = x.permute(1, 0, 2)
        # Local features
        #selected = [1,4,7,12]
        selected = [1,4,8,11]
        local_features = []
        for i in range(12):
            x = self.transformer.resblocks[i](x)
            if i in selected:
                #local_features.append(x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, grid, grid).contiguous().type(img.dtype))
                local_features.append(x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).contiguous().type(img.dtype))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return torch.cat(local_features,dim=1)

class ImageEmbddingLoss(torch.nn.Module):

    def __init__(self):
        super(ImageEmbddingLoss, self).__init__()
        self.model, _ = clip.load("ViT-B/32", device="cuda")
        self.model1, _1 = clip.load("RN101", device="cuda")
        self.model=self.model.eval().requires_grad_(False)
        self.model1=self.model1.eval().requires_grad_(False)
        self.transform = transforms.Compose([transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
        layers=torch.nn.Sequential(*list(self.model1.visual.children()))
        self.layer1=layers[:9]#[256 56 56]
        self.layer2=layers[:10]#[512 28 28]
        #self.layer3=layers[:11]#[1024 14 14]
        self.mse=torch.nn.MSELoss()
        
       
        
        self.cosloss = torch.nn.CosineEmbeddingLoss()
        self.features=CLIPVisualEncoder(self.model)
               
    def forward(self, masked_generated, masked_img_tensor):
        masked_generateds = self.face_pool(masked_generated)
        masked_generated_renormed = self.transform(masked_generateds * 0.5 + 0.5)
        masked_generated_feature1 = self.layer1(masked_generated_renormed.to(torch.float16))
        masked_generated_feature11 = self.layer2(masked_generated_renormed.to(torch.float16))
        #masked_generated_feature111 = self.layer3(masked_generated_renormed.to(torch.float16))
   
        masked_generateds=self.features(masked_generated)
        
        g1=masked_generateds[:,:768,:].contiguous()
        g2=masked_generateds[:,768:1536,:].contiguous()
        g3=masked_generateds[:,1536:2304,:].contiguous()
        g4=masked_generateds[:,2304:,:].contiguous()
        
        masked_img_tensors = self.face_pool(masked_img_tensor)
        masked_img_tensor_renormed = self.transform(masked_img_tensors * 0.5 + 0.5)
        masked_img_tensor_feature1 = self.layer1(masked_img_tensor_renormed.to(torch.float16))
        masked_img_tensor_feature11 =self.layer2(masked_img_tensor_renormed.to(torch.float16))
        #masked_img_tensor_feature111=self.layer3(masked_img_tensor_renormed.to(torch.float16))
        
        masked_img_tensors=self.features(masked_img_tensor)
        
        
        m1=masked_img_tensors[:,:768,:].contiguous()
        m2=masked_img_tensors[:,768:1536,:].contiguous()
        m3=masked_img_tensors[:,1536:2304,:].contiguous()
        m4=masked_img_tensors[:,2304:,:].contiguous()
        
        
        
        cos_target = torch.ones((masked_img_tensor.shape[0], 1)).float().cuda()
        
        loss=self.cosloss(g1,m1,cos_target).unsqueeze(0)+self.cosloss(g2,m2,cos_target).unsqueeze(0)+self.cosloss(g3,m3,cos_target).unsqueeze(0)+self.cosloss(g4,m4,cos_target).unsqueeze(0)
        t=self.mse(masked_generated_feature1, masked_img_tensor_feature1)+self.mse(masked_generated_feature11, masked_img_tensor_feature11)

        
        similarity = (loss + t)/2.0


        return similarity
