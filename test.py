def fun():
    for i in range(10):
        yield i
a = fun()
for item in a:
    print(item)