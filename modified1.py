import torch


class Conv2d_batchnorm(torch.nn.Module):

    def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = (1,1), activation = 'gelu'):
        super().__init__()
        self.activation = activation
        self.conv1 = torch.nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size, stride=stride, padding = 'same')
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        
        if self.activation == 'gelu':
            return torch.nn.functional.gelu(x)
        else:
            return x


class Multiresblock(torch.nn.Module):

    def __init__(self, num_in_channels, num_filters, alpha=1.67):
    
        super().__init__()
        self.alpha = alpha
        self.W = num_filters * alpha
        
        filt_cnt_3x3 = int(self.W*0.167)
        filt_cnt_5x5 = int(self.W*0.333)
        filt_cnt_7x7 = int(self.W*0.5)
        num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7
        
        self.shortcut = Conv2d_batchnorm(num_in_channels ,num_out_filters , kernel_size = (1,1), activation='None')

        self.conv_3x3 = Conv2d_batchnorm(num_in_channels, filt_cnt_3x3, kernel_size = (3,3), activation='gelu')

        self.conv_5x5 = Conv2d_batchnorm(filt_cnt_3x3, filt_cnt_5x5, kernel_size = (3,3), activation='gelu')
        
        self.conv_7x7 = Conv2d_batchnorm(filt_cnt_5x5, filt_cnt_7x7, kernel_size = (3,3), activation='gelu')

        self.batch_norm1 = torch.nn.BatchNorm2d(num_out_filters)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_out_filters)

    def forward(self,x):

        shrtct = self.shortcut(x)
        
        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)
        

        x = torch.cat([a,b,c],axis=1)        
        x = self.batch_norm1(x)

        x = x + shrtct
        x = self.batch_norm2(x)
        x = torch.nn.functional.gelu(x)
    
        return x


class Respath(torch.nn.Module):

    def __init__(self, num_in_filters, num_out_filters, respath_length):
    
        super().__init__()

        self.respath_length = respath_length
        self.shortcuts = torch.nn.ModuleList([])
        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])

        for i in range(self.respath_length):
            if(i==0):
                self.shortcuts.append(Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (1,1), activation='None'))
                self.convs.append(Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (3,3),activation='gelu'))

                
            else:
                self.shortcuts.append(Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation='None'))
                self.convs.append(Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation='gelu'))

            self.bns.append(torch.nn.BatchNorm2d(num_out_filters))
        
    
    def forward(self,x):

        for i in range(self.respath_length):

            shortcut = self.shortcuts[i](x)

            x = self.convs[i](x)
            x = self.bns[i](x)
            x = torch.nn.functional.gelu(x)

            x = x + shortcut
            x = self.bns[i](x)
            x = torch.nn.functional.gelu(x)

        return x


class MultiResUnet_Modified(torch.nn.Module):

    def __init__(self, input_channels=3, num_classes=1, alpha=1.67):
        super().__init__()
        
        self.alpha = alpha
        
        # Encoder Path
        self.multiresblock1 = Multiresblock(input_channels,32)
        self.in_filters1 = int(32*self.alpha*0.167)+int(32*self.alpha*0.333)+int(32*self.alpha* 0.5)
        self.pool1 =  torch.nn.MaxPool2d(2)
        self.respath1 = Respath(self.in_filters1,32,respath_length=4)

        self.multiresblock2 = Multiresblock(self.in_filters1,32*2)
        self.in_filters2 = int(32*2*self.alpha*0.167)+int(32*2*self.alpha*0.333)+int(32*2*self.alpha* 0.5)
        self.pool2 =  torch.nn.MaxPool2d(2)
        self.respath2 = Respath(self.in_filters2,32*2,respath_length=3)
    
    
        self.multiresblock3 =  Multiresblock(self.in_filters2,32*4)
        self.in_filters3 = int(32*4*self.alpha*0.167)+int(32*4*self.alpha*0.333)+int(32*4*self.alpha* 0.5)
        self.pool3 =  torch.nn.MaxPool2d(2)
        self.respath3 = Respath(self.in_filters3,32*4,respath_length=2)
    
    
        self.multiresblock4 = Multiresblock(self.in_filters3,32*8)
        self.in_filters4 = int(32*8*self.alpha*0.167)+int(32*8*self.alpha*0.333)+int(32*8*self.alpha* 0.5)
        self.pool4 =  torch.nn.MaxPool2d(2)
        self.respath4 = Respath(self.in_filters4,32*8,respath_length=1)
    
    
        self.multiresblock5 = Multiresblock(self.in_filters4,32*16)
        self.in_filters5 = int(32*16*self.alpha*0.167)+int(32*16*self.alpha*0.333)+int(32*16*self.alpha* 0.5)
     
        # Decoder path
        self.upsample6 = torch.nn.ConvTranspose2d(self.in_filters5,32*8,kernel_size=(2,2),stride=(2,2))  
        self.concat_filters1 = 32*8 *2
        self.multiresblock6 = Multiresblock(self.concat_filters1,32*8)
        self.in_filters6 = int(32*8*self.alpha*0.167)+int(32*8*self.alpha*0.333)+int(32*8*self.alpha* 0.5)

        self.upsample7 = torch.nn.ConvTranspose2d(self.in_filters6,32*4,kernel_size=(2,2),stride=(2,2))  
        self.concat_filters2 = 32*4 *2
        self.multiresblock7 = Multiresblock(self.concat_filters2,32*4)
        self.in_filters7 = int(32*4*self.alpha*0.167)+int(32*4*self.alpha*0.333)+int(32*4*self.alpha* 0.5)
    
        self.upsample8 = torch.nn.ConvTranspose2d(self.in_filters7,32*2,kernel_size=(2,2),stride=(2,2))
        self.concat_filters3 = 32*2 *2
        self.multiresblock8 = Multiresblock(self.concat_filters3,32*2)
        self.in_filters8 = int(32*2*self.alpha*0.167)+int(32*2*self.alpha*0.333)+int(32*2*self.alpha* 0.5)
    
        self.upsample9 = torch.nn.ConvTranspose2d(self.in_filters8,32,kernel_size=(2,2),stride=(2,2))
        self.concat_filters4 = 32 *2
        self.multiresblock9 = Multiresblock(self.concat_filters4,32)
        self.in_filters9 = int(32*self.alpha*0.167)+int(32*self.alpha*0.333)+int(32*self.alpha* 0.5)

        self.conv_final = Conv2d_batchnorm(self.in_filters9, num_classes+1, kernel_size = (1,1), activation='None')
        self.act = torch.nn.Sigmoid()
        self.conv_f = torch.nn.Conv2d(in_channels=51, out_channels=1, kernel_size=(1,1))


    def forward(self,x : torch.Tensor)->torch.Tensor:

        x_multires1 = self.multiresblock1(x)
        x_pool1 = self.pool1(x_multires1)
        x_multires1 = self.respath1(x_multires1)
        
        x_multires2 = self.multiresblock2(x_pool1)
        x_pool2 = self.pool2(x_multires2)
        x_multires2 = self.respath2(x_multires2)

        x_multires3 = self.multiresblock3(x_pool2)
        x_pool3 = self.pool3(x_multires3)
        x_multires3 = self.respath3(x_multires3)

        x_multires4 = self.multiresblock4(x_pool3)
        x_pool4 = self.pool4(x_multires4)
        x_multires4 = self.respath4(x_multires4)

        x_multires5 = self.multiresblock5(x_pool4)
                
        c1 = self.upsample6(x_multires5)
        up6 = torch.cat([c1,x_multires4],axis=1)
        up6_sum = c1 + x_multires4
        up6_sum = torch.cat([up6_sum,torch.zeros(up6_sum.shape)],axis=1)
        up6 = up6 + up6_sum  
        
        x_multires6 = self.multiresblock6(up6)
        
        c2 = self.upsample7(x_multires6)
        up7 = torch.cat([c2,x_multires3],axis=1)
        up7_sum = c2 + x_multires3
        up7_sum = torch.cat([up7_sum,torch.zeros(up7_sum.shape)],axis=1)
        up7 = up7 + up7_sum
        
        x_multires7 = self.multiresblock7(up7)

        c3 = self.upsample8(x_multires7)        
        up8 = torch.cat([c3,x_multires2],axis=1)
        up8_sum = c3 + x_multires2
        up8_sum = torch.cat([up8_sum,torch.zeros(up8_sum.shape)],axis=1)
        up8 = up8 + up8_sum
        
        x_multires8 = self.multiresblock8(up8)

        c4 = self.upsample9(x_multires8)
        up9 = torch.cat([c4,x_multires1],axis=1)
        
        up9_sum = c4 + x_multires1
        up9_sum = torch.cat([up9_sum,torch.zeros(up9_sum.shape)],axis=1)
        up9 = up9 + up9_sum
        
        x_multires9 = self.multiresblock9(up9)
        
        out =  self.conv_f(x_multires9)
        
        return self.act(out)
        

# Input_Image_Channels = 3
# def model() -> MultiResUnet_Modified:
#     model = MultiResUnet_Modified()
#     return model
# from torchsummary import summary
# model = model()
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256)])
