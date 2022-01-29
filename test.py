from math import sin, radians

for alpha in range(0,91):
    print(f"{alpha} = {(sin(radians(90-alpha))*sin(radians(45 -alpha/2)))/sin(radians(90 + alpha)/2)}")
