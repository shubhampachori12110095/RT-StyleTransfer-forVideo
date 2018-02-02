with open('flow.flo', 'rb') as f:
    a = f.readline()
    print(len(a))
    tag = a[:4]
    width = a[4:8]
    height = a[8:12]
    data = a[12:]
    print(int(width, 16))
    pass
