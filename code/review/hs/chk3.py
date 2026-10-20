from scipy.stats import norm
for q in [0.9,0.75,0.5,0.25]:
    c=norm.ppf(1-q); l=norm.pdf(c)/q; d=l*(l-c)
    row=[q,round(d,3)]
    for R in [0.1,0.5,0.65]:
        f=(1-d)/(1-d*R); row+= [round((1-d)**2/(1-d*R),3), round(f*f,3)]
    print(row)
d=2/3.141592653589793
for R in [0.5,0.65]: print(R,(1-d)/(1-d*R),(1-d)**2/(1-d*R),((1-d)/(1-d*R))**2)
