# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 19:34:41 2023

@author: ICML4119 AUTHORS
"""

from myimports import *
  

def train(trainloader, model, criterion, optimizer, epoch,args):
    model.train()
    total_loss = 0
   
    
    print(f'mic:{args.max_fic}')
    
    for batch_id, (feats, label) in enumerate(trainloader):
  
        feats = feats.cuda()
        label = label.cuda()
        
        # print(feats.shape)
        
        # window-based argment
        if args.dropout_patch>0:
            selecy_window_indx = random.sample(range(10),int(args.dropout_patch*10))
            inteval = int(len(feats)//10)
            
            for idx in selecy_window_indx:
                feats[:,idx*inteval:idx*inteval+inteval,:] = torch.randn(1).cuda()
        
   

        optimizer.zero_grad()
        prediction  = model(feats)
        
       
        loss = criterion(prediction, label)
       
        if True:
            loss = loss 
            sys.stdout.write('\r Training batch [%d/%d] batch loss: %.4f  total loss: %.4f' % \
                            (batch_id, len(trainloader), loss.item(),loss.item()))
        
        
     
        loss.backward()
        
    
        
        
        if args.max_fic <0.0000000001: ## turn fic off
    
            pass
        else:
            FIM_constraint_(model.parameters(), args.max_fic) #apply fic
        
        optimizer.step()
        

        total_loss = total_loss + loss
        

    return total_loss / len(trainloader)



def test(testloader, model, criterion, args):
    model.eval()

    total_loss = 0
    test_labels = []
    test_predictions = []
    
    with torch.no_grad():

        for batch_id, (feats, label) in enumerate(testloader):
  
            feats = feats.cuda()
            label = label.cuda()
         
            prediction = model(feats)  #b*class
           

            loss = criterion(prediction,label)
            
            
            
            loss = loss
            total_loss = total_loss + loss.item()

            sys.stdout.write('\r Testing batch [%d/%d] batch loss: %.4f' % (batch_id, len(testloader), loss.item()))
            
            # test_labels.extend([label.squeeze().cpu().numpy()])
            test_labels.extend([label.cpu().numpy()])
            test_predictions.extend([(prediction).cpu().numpy()])
    
    test_labels = np.vstack(test_labels)
    test_predictions = np.vstack(test_predictions)

    test_predictions_prob = np.exp(test_predictions)/np.sum(np.exp(test_predictions),axis=1,keepdims=True)
    
    
    test_predictions = np.argmax(test_predictions,axis=1)
    test_labels = np.argmax(test_labels,axis=1)
    avg_score = accuracy_score(test_labels,test_predictions)
    balanced_avg_score = balanced_accuracy_score(test_labels,test_predictions)
    f1_marco = f1_score(test_labels,test_predictions,average='macro')
    f1_micro = f1_score(test_labels,test_predictions,average='micro')
    
    p_marco = precision_score(test_labels,test_predictions,average='macro')
    p_micro = precision_score(test_labels,test_predictions,average='micro')
    
    r_marco = recall_score(test_labels,test_predictions,average='macro')
    r_micro = recall_score(test_labels,test_predictions,average='micro')
    
    
    r_marco = recall_score(test_labels,test_predictions,average='macro')
    r_micro = recall_score(test_labels,test_predictions,average='micro')
    
    
    #### not used auu
    if args.num_classes ==2: 
        # print(test_labels.shape)
        
        roc_auc_ovo_marco=  0.# roc_auc_score(test_labels,test_predictions_prob[:,1],average='macro')
        roc_auc_ovo_micro=  0.# roc_auc_score(test_labels,test_predictions_prob,average='micro',multi_class='ovo')
    
        roc_auc_ovr_marco=  0.# roc_auc_score(test_labels,test_predictions_prob[:,1],average='macro')
        roc_auc_ovr_micro=  0.# roc_auc_score(test_labels,test_predictions_prob,average='micro',multi_class='ovr')
    
    
    else:
        roc_auc_ovo_marco=  0.# roc_auc_score(test_labels,test_predictions_prob,average='macro',multi_class='ovo')
        roc_auc_ovo_micro=  0.# not used ! roc_auc_score(test_labels,test_predictions_prob,average='micro',multi_class='ovo')

        roc_auc_ovr_marco=  0.#roc_auc_score(test_labels,test_predictions_prob,average='macro',multi_class='ovr')
        roc_auc_ovr_micro=  0.# not used
    

    results = [avg_score,balanced_avg_score,f1_marco,f1_micro, p_marco,p_micro,r_marco,r_micro,roc_auc_ovo_marco,roc_auc_ovo_micro,roc_auc_ovr_marco,roc_auc_ovr_micro]
    
    
    return total_loss / len(testloader), results



def FIM_constraint_(weights, max_fic, norm_type=2.0):
    
    # simple imepelmentation of fisher information constrant
    
    if isinstance(weights, torch.Tensor):
        weights = [weights]
    weights = [p for p in weights if p.grad is not None]
    max_fic = float(max_fic)
    norm_type = float(norm_type)
    if len(weights) == 0:
        return torch.tensor(0.)
    
    device = weights[0].grad.device
    
    # Calculate the |F|_1 with approximation
    fisher_total = torch.sum(torch.stack([torch.norm(p.grad.detach(), norm_type).pow(2).to(device) for p in weights]))
    
   
 
    # Compute the renormalize coefficient based on the fisher_total
    renorm_coef = max_fic / (fisher_total + 1e-6)
    
    
    if renorm_coef < 1: #need renorm
        renorm_coef = torch.sqrt(renorm_coef)
        for p in weights:
            p.grad.detach().mul_(renorm_coef.to(p.grad.device))
    
    # Return the fisher information before re-norm
    return fisher_total

def main():
    parser = argparse.ArgumentParser(description='time classification by FIC-TSC')
    parser.add_argument('--dataset', default="SelfRegulationSCP1", type=str, help='dataset SelfRegulationSCP1')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers used in dataloader [4]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512] resnet-50 1024')
    parser.add_argument('--lr', default=5e-3, type=float, help='1e-3 Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay 1e-4]')
    parser.add_argument('--dropout_patch', default=0.5, type=float, help='window dropout rate [0] 0.5')
    parser.add_argument('--dropout_node', default=0.2, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--seed', default='0', type=int, help='random seed')
   
    parser.add_argument('--optimizer', default='adamw', type=str, help='adamw sgd')
    
    parser.add_argument('--save_dir', default='./savemodel/', type=str, help='the directory used to save all the output')
    parser.add_argument('--epoch_des', default=10, type=int, help='turn on warmup')
   
    parser.add_argument('--embed', default=128, type=int, help='Number of embedding')
    
    parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
    parser.add_argument('--max_fic', default=2., type=float, help='max fim')
    
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    args.save_dir = args.save_dir+'InceptBackbone'
    
    maybe_mkdir_p(join(args.save_dir, f'{args.dataset}'))
    args.save_dir = make_dirs(join(args.save_dir, f'{args.dataset}'))
    maybe_mkdir_p(args.save_dir)
    


    # <------------- set up logging ------------->
    logging_path = os.path.join(args.save_dir, 'Train_log.log')
    logger = get_logger(logging_path)

    # <------------- save hyperparams ------------->
    option = vars(args)
    file_name = os.path.join(args.save_dir, 'option.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(option.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')



    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
 
    
    ###################################
    if args.dataset in ['JapaneseVowels','SpokenArabicDigits','CharacterTrajectories','InsectWingbeat']:

        trainset = loadorean(args, split='train')
        testset = loadorean(args, split='test')
        
        seq_len,num_classes,L_in=trainset.max_len,trainset.num_class,trainset.feat_in
        
        
        
        
        print(f'max lenght {seq_len}')
        args.feats_size = L_in
        args.num_classes =  num_classes
        print(f'num class:{args.num_classes}' )
        
    else:
        Xtr, ytr, meta = load_classification(name=args.dataset,split='train')

        print(Xtr.shape)
        word_to_idx = {}
        for i in range(len(meta['class_values'])):
            word_to_idx[meta['class_values'][i]]=i

        Xtr =torch.from_numpy(Xtr).permute(0,2,1).float()
        ytr = [word_to_idx[i] for i in ytr]
        ytr =  F.one_hot(torch.tensor(ytr)).float()
        print(ytr.shape)

        trainset = TensorDataset(Xtr,ytr)

        Xte, yte, _ = load_classification(name=args.dataset,split='test')

        Xte =torch.from_numpy(Xte).permute(0,2,1).float()
        yte = [word_to_idx[i] for i in yte]
        yte = F.one_hot(torch.tensor(yte)).float()


        testset = TensorDataset(Xte,yte)




        args.feats_size = Xte.shape[-1]
        L_in = Xte.shape[-1]
        # print(L_in)
        num_classes = yte.shape[-1]
        args.num_classes =  yte.shape[-1]
        print(f'num class:{args.num_classes}' )
        # args.num_classes = num_classes

        seq_len=  max(21, Xte.shape[1])
    
    
    
    
    # <------------- define  network ------------->
   
    
    model = ItimeNet(args.feats_size,mDim=args.embed,n_classes =num_classes,dropout=args.dropout_node, max_seq_len = seq_len).cuda()
    
    
    
    
    if  args.optimizer == 'adamw':
    
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
        optimizer =Lookahead(optimizer)    
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        # optimizer =Lookahead(optimizer) 
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer =Lookahead(optimizer) 
    
    elif args.optimizer == 'adamp':
        optimizer = AdamP(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer =Lookahead(optimizer) 
    
    
    ####split train and vali
    
    total_size = len(trainset)
    val_ratio = 0.2  # for 20% validation

    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    train_subset, val_subset = random_split(trainset, [train_size, val_size])
    print(f'total_size:{total_size}')
    print(f'train_subset:{len(train_subset)}')
    # trainloader = DataLoader(trainset, args.batchsize, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    trainloader = DataLoader(train_subset, args.batchsize, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    valiloader = DataLoader(val_subset, 128, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    testloader = DataLoader(testset, 128, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    

    best_score = 0
    save_path = join(args.save_dir, 'weights')
    os.makedirs(save_path, exist_ok=True)
    

    results_best = None
    for epoch in range(1, args.num_epochs + 1):

        train_loss_bag = train(trainloader, model, criterion, optimizer, epoch,args) # iterate all bags
        
        
        vali_loss,results= test(valiloader, model, criterion, args)
 
        [avg_score_vali,balanced_avg_score,f1_marco,f1_micro,p_marco,p_micro,r_marco,r_micro,roc_auc_ovo_marco,roc_auc_ovo_micro,roc_auc_ovr_marco,roc_auc_ovr_micro] = results
        
        
        logger.info('\r validating: Epoch [%d/%d] train loss: %.4f test loss: %.4f, accuracy: %.4f, bal. average score: %.4f, f1 marco: %.4f   f1 mirco: %.4f  p marco: %.4f   p mirco: %.4f r marco: %.4f   r mirco: %.4f  roc_auc ovo marco: %.4f   roc_auc ovo mirco: %.4f  roc_auc ovr marco: %.4f   roc_auc ovr mirco: %.4f' % 
                  (epoch, args.num_epochs, train_loss_bag, vali_loss, avg_score_vali,balanced_avg_score,f1_marco,f1_micro, p_marco,p_micro,r_marco,r_micro,roc_auc_ovo_marco,roc_auc_ovo_micro,roc_auc_ovr_marco,roc_auc_ovr_micro )) 
        
        
        test_loss,test_results= test(testloader, model, criterion, args)
 
        [avg_score,balanced_avg_score,f1_marco,f1_micro,p_marco,p_micro,r_marco,r_micro,roc_auc_ovo_marco,roc_auc_ovo_micro,roc_auc_ovr_marco,roc_auc_ovr_micro] = test_results
        
        
        logger.info('\r testing: Epoch [%d/%d] train loss: %.4f test loss: %.4f, accuracy: %.4f, bal. average score: %.4f, f1 marco: %.4f   f1 mirco: %.4f  p marco: %.4f   p mirco: %.4f r marco: %.4f   r mirco: %.4f  roc_auc ovo marco: %.4f   roc_auc ovo mirco: %.4f  roc_auc ovr marco: %.4f   roc_auc ovr mirco: %.4f' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss, avg_score,balanced_avg_score,f1_marco,f1_micro, p_marco,p_micro,r_marco,r_micro,roc_auc_ovo_marco,roc_auc_ovo_micro,roc_auc_ovr_marco,roc_auc_ovr_micro )) 
        
          



if __name__ == '__main__':
    main()