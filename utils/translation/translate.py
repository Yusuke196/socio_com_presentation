from transformers import pipeline

fugu_translator = pipeline('translation', model='staka/fugumt-en-ja', device='cuda:7')
print(fugu_translator(['This is a cat.', 'This is not a cat.']))
