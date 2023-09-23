from DecisionTree.id3 import id3

label_values = ['unacc', 'acc', 'good', 'vgood']
attributes = [
    ('buying', ['vhigh', 'high', 'med', 'low'],)
    ('maint', ['vhigh', 'high', 'med', 'low'],)
    ('doors', [2, 3, 4, '5more'],)
    ('persons', [2, 4, 'more'],)
    ('lug_boot', ['small', 'med', 'big'],)
    ('safety', ['low', 'med', 'high'],)
]

x = []
y = []

with open("train.csv") as f:
    for line in f:
        values = line.strip().split(',')
        x.append(values[:-1])  # add all independent variables to x
        y.append(values[-1])  # add label to y

print("TODO: id3()")