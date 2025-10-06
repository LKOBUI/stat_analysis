'''
Example 8.3:
Let y be the total electricity consumption (in kilowatt-hours)
x1 be the size of the house (square feet of floor space)
four types of air conditioning systems:
no air conditioning        -> NAC  [d]
window units               -> WU   [a]
heat pump                  -> HP   [b]
central air conditioning   -> CAC  [c]
Let Data:
Y   X1   ACType
           a
           a
           b
           a
           c
           c
           d
           d
           d
           a
           b
           a
           b
           c
           d
           d
           c
           c
           a
           b
           c
Qus: How to write eqn for you models ?
Ans: you need to take only three variable 
     in your modles. And forth variable is define
     itself. So in above example i will take only
     three variale i.e. a,b,c. d variable self define by your models
     See how 

Y = β0 + β1X1 + β2a + β3b + β4c + β5X1a + β6X1b + β7X1c +  ε    (i)
In above example ihave not written 4th var i.e. d.
Now See

house with No AC eqn (i) become
    No AC -> put a =0, b =0, c= 0 in eqn(i)    
Y  = β0 + β1X1   (ii)    -> Eqn represent house have no AC

house with window units eqn (i) become
    WU -> put a =1, b =0, c= 0 in eqn(i) 
Y = (β0 + β2) + (β1 + β5)X1 + ε

house with heat pump eqn (i) become
    HP -> put a =0, b =1, c= 0 in eqn(i)
Y = (β0 + β3) + (β1 + β6)X1 + ε

house with central air conditioning eqn (i) become
    CAC -> put a =0, b =0, c= 1 in eqn(i)
Y = (β0 + β4) + (β1 + β7)X1 + ε
'''