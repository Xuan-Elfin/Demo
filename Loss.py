loss += nn.MSELoss()(self.normalize(w),W[0][i])                               # 互信息最大化lo
loss += nn.L1Loss()(w,W[0])               
loss += nn.SmoothL1Loss()(w,W[0])       
loss += torch.mean(F.binary_cross_entropy_with_logits((w.to(A.device)),(W[0][i]))) 
loss += nn.PoissonNLLLoss()(w,W[0])        
loss += nn.KLDivLoss()(w,W[0])           
loss += nn.HingeEmbeddingLoss()(w,W[0])  
loss += nn.SoftMarginLoss()(w,W[0])    

loss += nn.TripletMarginLoss()(w,W[0])
loss += nn.CTCLoss()(w,W[0])
loss += nn.MarginRankingLoss(margin=0)(w,W[0])
loss += nn.MarginRankingLoss(margin=0)(w,W[0])
loss += nn.CosineEmbeddingLoss()(w,W[0])
