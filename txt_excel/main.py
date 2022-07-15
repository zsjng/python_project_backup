with open('test.txt','r',encoding='utf-8') as f:
    list_1 = list(f)
a = ''
# print(list_1)
for i in list_1:
    i = i.replace('!!!!!',',')
    # i = i.replace('positive mean ','positive')
    # i = i.replace('number positives ', 'np')
    # i = i.replace('negative mean ', 'negetive')
    # i = i.replace('SHAP values for ll: mean-', 'll')
    a = i.replace('].','],')

b = a.split(',')
#print(b,type(b))
j = 0
datalist = []
while j < len(b)-24:
    data = []
    for i in range(j,j+24):
        b[i] = b[i].strip()
        print(b[i],end=',')
        data.append(b[i].strip())
        #print(data)
    j += 24
    print('\n')
