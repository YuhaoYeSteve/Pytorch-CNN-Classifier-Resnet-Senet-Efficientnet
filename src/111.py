# import torch


# # torch.where(condition, x, y) → Tensor
# predict = torch.tensor([1, 1, 1, 0])
# gt = torch.tensor([1, 1, 1, 1])
# print("predict", predict)
# print("gt", gt)

# right = (predict==gt).to(torch.int64)
# # wrong = torch.where(right==0, right, gt)
# # print(right)
# # print(wrong)
# wrong = torch.eq(right, 0)
# print(wrong)
# # print(torch.where(predict==gt, predict, gt))


a = [1, 2, 3, 1]
print(list(enumerate(a)))
