import re
word_list = {

  # place your extracted and replaceable common nouns in the form of a dictionary here:
  # 'original-noun' : 'replaceable-noun'
              }
with open('./gold-pmb-original/train-original.txt.raw') as main, open(
        'noun_augmentation_train_files/Text-noun-changed-inside-columns_with_SS.txt', 'w') as done:
     text = main.read()
     done.write(re.sub(r'\b\w+\b', lambda x: word_list.get(x.group(), x.group()), text))
