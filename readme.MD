use l to switch control to virtual-screen location , arrows keys will change VS locaiton
use s to switch control to virtual-screen size , arrows keys will change VS size 
supported keyboard actions 
    ? = for help 
    l - switch to virtual screen location setting
    s - switch to virtual screen size setting 
    <Up> - move up or reduce virtual screen height
    <Down> - move down or extend virtual screen hegith
    <Left>  - move left or reduce virtual screen width
    <Right> - move right or extend virtual screen size 
    u - to update app config with current virtual screen location

    r - run the simulation, pressing r again will restart the simulaiton from the begining and reload the latest version of the CSV
    p - pause the simulation 

    e - to exit


appConfig.csv
fishScreenStartX
fishScreenStartY
fishScreenWidth
fishScreenHeight
f9CommunicationEnabled - if on then we try to communicate with a server running on the Stream-7 machine (needs to be up and running). Note that if the server is not running then the program will hangs as the communication is synchronous 

stimulusConfig.csv
The coordinates of the virtual screen are between 0-999. top left corenr is (0,0)
We automatically translates these coordinates to the real size of the virtual screen, which might change depending on the setup
example : if the real virtual screensize is 100x50 then all coordinates between (0,0) and (99,49) will be mapped to (0,0)
the formula is trunc(x/virtual screen width) and trunc(y/virtual screen height)

Shape	: Circle, Rectangle 
ShapeWidth : 
ShapeHeight : ignored for circle
StartX : x=0 is the top-left corner of the fishScreen as defined in appConfig.csv fishScreen parameters 
StartY : y=0 is the top-left corner of the fishScreen as defined in appConfig.csv fishScreen parameters 
EndX : 
EndY :
repetition 
fastSpeed/slowSpeed = milisecond, time waited before the stimulus is moved to the next x,y
fastDuration/slowDuration = how many cycles of fast/speed before switching to the other one
