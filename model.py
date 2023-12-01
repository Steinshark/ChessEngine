import torch 


class ChessModel(torch.nn.Module):

    def __init__(self,in_ch):

        super(ChessModel,self).__init__()

        self.v_conv_n      = 32
        self.h_conv_n      = 32
        self.q_conv_n      = 32

        self.conv_act       = torch.nn.functional.leaky_relu
        self.lin_act        = torch.nn.functional.relu
        self.softmax        = torch.nn.functional.softmax

        self.vert_conv1     = torch.nn.Conv2d(in_ch,self.v_conv_n,kernel_size=(8+8+1,1),stride=1,padding=(8,0))
        self.horz_conv1     = torch.nn.Conv2d(in_ch,self.h_conv_n,kernel_size=(1,8+8+1),stride=1,padding=(0,8))
        self.quad_conv1     = torch.nn.Conv2d(in_ch,self.q_conv_n,kernel_size=(7),stride=1,padding=3)

        self.vert_conv2     = torch.nn.Conv2d(self.v_conv_n+self.h_conv_n+self.q_conv_n,self.v_conv_n,kernel_size=(8+8+1,1),stride=1,padding=(8,0))
        self.horz_conv2     = torch.nn.Conv2d(self.v_conv_n+self.h_conv_n+self.q_conv_n,self.h_conv_n,kernel_size=(1,8+8+1),stride=1,padding=(0,8))
        self.quad_conv2     = torch.nn.Conv2d(self.v_conv_n+self.h_conv_n+self.q_conv_n,self.q_conv_n,kernel_size=(7),stride=1,padding=3)

        self.vert_conv3     = torch.nn.Conv2d(self.v_conv_n+self.h_conv_n+self.q_conv_n,self.v_conv_n,kernel_size=(8+8+1,1),stride=1,padding=(8,0))
        self.horz_conv3     = torch.nn.Conv2d(self.v_conv_n+self.h_conv_n+self.q_conv_n,self.h_conv_n,kernel_size=(1,8+8+1),stride=1,padding=(0,8))
        self.quad_conv3     = torch.nn.Conv2d(self.v_conv_n+self.h_conv_n+self.q_conv_n,self.q_conv_n,kernel_size=(7),stride=1,padding=3)

        self.flatten        = torch.nn.Flatten()

        self.linear1        = torch.nn.Linear(6144,1024)
        self.linear2        = torch.nn.Linear(1024,256)
        self.linear3        = torch.nn.Linear(256,1)

            


    def forward(self,x:torch.Tensor) -> torch.Tensor:
        
        #ITER1 Get vertical, horizontal, and square convolutions 
        vert_convolutions1  = self.conv_act(self.vert_conv1(x))                                         #Out    = (32,8,8)
        horz_convolutions1  = self.conv_act(self.horz_conv1(x))                                         #Out    = (32,8,8)
        quad_convolutions1  = self.conv_act(self.quad_conv1(x))                                         #Out    = (32,8,8)
        comb_convolutions1  = torch.cat([vert_convolutions1,horz_convolutions1,quad_convolutions1],dim=1)

        vert_convolutions2  = self.conv_act(self.vert_conv2(comb_convolutions1))                        #Out    = (128,8,8)
        horz_convolutions2  = self.conv_act(self.horz_conv2(comb_convolutions1))                        #Out    = (128,8,8)
        quad_convolutions2  = self.conv_act(self.quad_conv2(comb_convolutions1))                        #Out    = (128,8,8)
        comb_convolutions2  = torch.cat([vert_convolutions2,horz_convolutions2,quad_convolutions2],dim=1)

        vert_convolutions3  = self.conv_act(self.vert_conv3(comb_convolutions2))                        #Out    = (128,8,8)
        horz_convolutions3  = self.conv_act(self.horz_conv3(comb_convolutions2))                        #Out    = (128,8,8)
        quad_convolutions3  = self.conv_act(self.quad_conv3(comb_convolutions2))                        #Out    = (128,8,8)
        comb_convolutions3  = torch.cat([vert_convolutions3,horz_convolutions3,quad_convolutions3],dim=1)

        x                   = self.flatten(comb_convolutions3)
        x                   = self.lin_act(self.linear1(x))
        x                   = self.lin_act(self.linear2(x))
        x                   = torch.nn.functional.tanh(self.linear3(x))
        


        return x     



if __name__ == "__main__":

    m       = ChessModel(3).to(torch.device('cuda'))

    inv     = torch.randn(size=(78,3,8,8),device=torch.device('cuda'))

    y       = m.forward(inv)

    print(f"out is {y.shape}")
