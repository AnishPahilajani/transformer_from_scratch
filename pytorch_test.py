import torch
import torch.nn as nn
import math

# Assuming your original tensor has shape (1, 5, 12)
h = 4
d_model = 12
d_k = int(d_model//h) # 64 
seq_len = 5

original_tensor = torch.randn(1, seq_len, d_model)

# Reshape the tensor to (1, 8, 5, 64)
reshaped_tensor = original_tensor.view(1, h, seq_len, d_k)
print(reshaped_tensor.shape)


a = original_tensor.view(original_tensor.shape[0], original_tensor.shape[1], h, d_k).transpose(1, 2)
print(a.shape)
# print(reshaped_tensor[0])
# print(a[0])
print(torch.equal(a, reshaped_tensor))

#print(pe[:, : 512].shape)
# x = torch.randn(1, 512)
# print(x.size(1))



# original_tensor = torch.tensor([[[1, 2, 3, 4],
#                                  [5, 6, 7, 8],
#                                  [9, 10, 11, 12]],
#                                 [[13, 14, 15, 16],
#                                  [17, 18, 19, 20],
#                                  [21, 22, 23, 24]]])

# print("Original Tensor:")
# print(original_tensor)
# print("Shape:", original_tensor.shape)

# # Transpose dimensions 1 and 2
# transposed_tensor = original_tensor.transpose(1, 2)

# print("\nTransposed Tensor:")
# print(transposed_tensor)
# print("Shape:", transposed_tensor.shape)





# VV IMP
# understanding: 
# (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
# query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)


original_tensor = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                 [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]],
                                [[25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
                                 [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]]])

print("Original Tensor:")
print(original_tensor)
print("Shape:", original_tensor.shape)

# Convert 3D tensor to 4D tensor with shape (2=batch, 2=seq, h=3, 4=d_k)
reshaped_tensor = original_tensor.view(2, 2, 3, 4)

print("\nReshaped Tensor:")
print(reshaped_tensor)
print("Shape:", reshaped_tensor.shape)

reshaped_tensor2 = reshaped_tensor.transpose(1, 2) #(2=batch,  h=3, 2=seq, 4=d_k)

print("\nReshaped Tensor2:")
print(reshaped_tensor2)
print("Shape2:", reshaped_tensor2.shape)

print("_____________ATTENTION MATRIX________________")
key_ = reshaped_tensor2.transpose(-2, -1)
print("key shape ", key_.shape)
# print(key_)
print("attention matrix shape: ", (reshaped_tensor2 @ key_).shape)
att_mat = reshaped_tensor2 @ key_
print(att_mat)

print("_____________ mult with VALUE __________")
final_att = att_mat @ reshaped_tensor2
print("mult with value: ",final_att.shape)
print(final_att)


print("_____________FINAL CONCATNATE __________")
f = final_att.transpose(1, 2).contiguous().view(final_att.shape[0], -1, 3*4)
print(f.shape)
print(f)