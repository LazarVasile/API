def shapeArea(n):
    sum = 0
    for i in range(1, n):
        print(i)
        sum = sum + pow(2, i+1)
    
    return sum

print(shapeArea(4))