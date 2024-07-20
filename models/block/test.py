import torch

# x = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
first = torch.rand(32,50,64,64)
x = first
x = torch.split(x, 8, dim = 3)

# print(x[0].shape)
z = [x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]]
for i in range(0,8):
    y = torch.split(x[i], 8, dim = 2)
    # for j in range(0,8):
        
    #     # print(y[j].shape) 此y[j]torch.Size([16, 50, 8, 8]) 直接做attention
    z[i] = torch.cat((y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7]),dim = 2, out = None)
    # print(z[i].shape)
res1 = torch.cat((z[0],z[1],z[2],z[3],z[4],z[5],z[6],z[7]),dim = 3, out = None)
# print(res.shape)
print(res1.equal(first))
x = first
x = torch.split(x, 8, dim = 3)

# print(x[0].shape)
c = [x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]]
for i in range(0,8):
    y = torch.split(x[i], 8, dim = 2)
    z = [y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7]]
    for j in range(0,8):
        # print(y[j].shape)
        z[j] = y[j].reshape(32,3200,1,1)
        # print(z[j].shape)
    c[i] = torch.cat((z[0],z[1],z[2],z[3],z[4],z[5],z[6],z[7]),dim = 2, out = None)
    # print(c[i].shape)
attready = torch.cat((c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]),dim = 3, out = None)
# print(attready.shape)
# 做attention
att = torch.split(attready, 1, dim = 3)
for i in range(0,8):
    ori = torch.split(att[i], 1, dim = 2)
    # print(ori)
    for j in range(0,8):
        z[j]=ori[j].reshape(32,50,8,8)
        # print(z[j].shape)
        # print(z[j])
    c[i] = torch.cat((z[0],z[1],z[2],z[3],z[4],z[5],z[6],z[7],),dim = 2, out = None)
res2 = torch.cat((c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]),dim = 3, out = None)

# res2 = attready.reshape(16,50,64,64)
print(res2.shape)
print(res2.equal(first))




# x = torch.split(x, 8, dim = 1)
# print(x[0])
# y = torch.split(x[0], 2, dim = 2)
# print(y[0])
# z = y[0].reshape(8,1,1)
# print(z)
# print(z.shape)
        
# print(x[0][0].shape)

# y = torch.split(x[0], 2, dim = 0)
# z = torch.split(x[1], 2, dim = 0)
# print(y[0])

# f1 = torch.cat((y[0],y[1]),dim = 0, out = None)
# f2 = torch.cat((z[0],z[1]),dim = 0, out = None)
# f = torch.cat((f1,f2),dim = 1, out = None)
# print(f)
# x = torch.tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]])
# # x = torch.rand(16,50,64,64)
# # print(x.shape)
# # x = torch.split(x, 2, dim = 1)
# # print(x[0])
# # y = torch.split(x[0], 2, dim = 2)
# # print(y[0])
# # z = y[0].reshape(8,1,1)
# # print(z)
# # print(z.shape)
# first = x


# x = torch.split(x, 2, dim = 1)

# # print(x[0].shape)
# c = [x[0],x[1]]
# for i in range(0,2):
#     y = torch.split(x[i], 2, dim = 2)
#     z = [y[0],y[1]]
#     for j in range(0,2):
#         # print(y[j].shape)
#         z[j] = y[j].reshape(8,1,1)
#         # print(z[j])
#     c[i] = torch.cat((z[0],z[1]),dim = 2, out = None)
#     # print(c[i])
# attready = torch.cat((c[0],c[1]),dim = 1, out = None)
# # 做attention
# # print(attready.shape)
# # print(attready)
# att = torch.split(attready, 1, dim = 2)
# for i in range(0,2):
#     ori = torch.split(att[i], 1, dim = 1)
#     # print(ori)
#     for j in range(0,2):
#         z[j]=ori[j].reshape(2,2,2)
#         # print(z[j].shape)
#         # print(z[j])
#     c[i] = torch.cat((z[0],z[1]),dim = 1, out = None)
# res2 = torch.cat((c[0],c[1]),dim = 2, out = None)


# # print(att[0])
# # 做attention
# # res2 = attready.reshape(2,4,4)
# print(res2.shape)
# print(res2)
# print(res2.equal(first))