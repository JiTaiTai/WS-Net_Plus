import timm
import torch
import torch.nn as nn


class Efficientnetv2(nn.Module):
    def __init__(self, name, pretrained=True):
        super().__init__()
        if name.startswith("tf_efficientnetv2_s_in21k"):  
            self.extract = timm.create_model('tf_efficientnetv2_s_in21k', features_only=True,
                                             out_indices=(1, 2, 3, 4), pretrained=False)
        elif name.startswith("tf_efficientnetv2_s_in21ft1k"):
            self.extract = timm.create_model('tf_efficientnetv2_s_in21ft1k', features_only=True,
                                             out_indices=(1, 2, 3, 4), pretrained=pretrained)
        elif name.startswith("efficientnetv2_rw_s"):
            self.extract = timm.create_model('efficientnetv2_rw_s', features_only=True,
                                             out_indices=(1, 2, 3, 4), pretrained=pretrained)
        elif name.startswith("efficientnetv2_rw_m"):
            self.extract = timm.create_model('efficientnetv2_rw_m', features_only=True,
                                             out_indices=(1, 2, 3, 4), pretrained=pretrained)
        elif name.startswith("tf_efficientnetv2_l_in21ft1k"):
            self.extract = timm.create_model('tf_efficientnetv2_l_in21ft1k', features_only=True,
                                             out_indices=(1, 2, 3, 4), pretrained=pretrained)
        else:
            raise Exception("Error, please check the backbone name!")

        if pretrained:
            print("==> Load pretrained model for: {} successfully".format(name))

    def forward(self, x):
        f1, f2, f3, f4 = self.extract(x)

        return f1, f2, f3, f4

class ResNet50(nn.Module):
    def __init__(self, name, pretrained=True):
        super().__init__()
        if name.startswith("resnet50"):   
            # self.extract = timm.create_model('resnext101_32x4d', features_only=True,
            #                                  out_indices=(1, 2, 3, 4), pretrained=False)
            # self.extract = timm.create_model('resnet50', features_only=True,
            #                                  out_indices=(1, 2, 3, 4), pretrained=False)
            self.extract = timm.create_model('resnet18', features_only=True,
                                              out_indices=(1, 2, 3, 4), pretrained=True)
        else:
            raise Exception("Error, please check the backbone name!")

        if pretrained:
            print("==> Load pretrained model for: {} successfully".format(name))

    def forward(self, x):
        f1, f2, f3, f4 = self.extract(x)

        return f1, f2, f3, f4


if __name__ == "__main__":
    # model_names = timm.list_models("*eff*v2*s*", pretrained=True)
    # for name in model_names:
    #     print(name)
    model = Efficientnetv2('efficientnetv2_rw_s')
    f1, f2, f3, f4 = model(torch.randn(2, 3, 512, 512))
    for x in (f1, f2, f3, f4):
        print(x.shape)
