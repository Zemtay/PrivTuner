# from packet import *
#
# def test(models,device,dataloader,epoch): #find the best model
#
#     accuracies = []
#     for (index, model) in enumerate(models):
#         model.eval()
#         acc = 0
#         for batch in tqdm(dataloader, desc=f"Testing Epoch {epoch} model{index+1}:"):
#                 inputs, targets = [x.to(device) for x in batch]
#                 with torch.no_grad():
#                     bert_output = model(inputs)
#                     acc += (bert_output.argmax(dim=1) == targets).sum().item()
#                 print()
#         print(f"Acc: {acc / len(dataloader):.2f}")
#         accuracies.append(acc)
#         bestmodel_index = np.argmax(accuracies)
#     return bestmodel_index