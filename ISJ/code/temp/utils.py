
def load_split_eunki_data(total_train_dataset, total_train_label, train_idx, val_idx):

  train_dataset= total_train_dataset.iloc[train_idx]
  val_dataset= total_train_dataset.iloc[val_idx]
  train_label = total_train_label.iloc[train_idx]
  val_label = total_train_label.iloc[val_idx]

  train_dataset.reset_index(drop= True, inplace= True)
  val_dataset.reset_index(drop= True, inplace= True)
  train_label.reset_index(drop= True, inplace= True)
  val_label.reset_index(drop= True, inplace= True)
        
  temp = []    
        
  for val_idx in val_dataset.index:
    if val_dataset['is_duplicated'].iloc[val_idx] == True:      
      if val_dataset['sentence'].iloc[val_idx] in train_dataset['sentence'].values:
        train_dataset.append(val_dataset.iloc[val_idx])
        train_label.append(val_label.iloc[val_idx])
        temp.append(val_idx)
                
  val_dataset.drop(temp, inplace= True, axis= 0)
  val_label.drop(temp, inplace= True, axis= 0)

  train_label_list = train_label['label'].values.tolist()
  val_label_list = val_label['label'].values.tolist()
  
  return train_dataset, val_dataset, train_label_list, val_label_list