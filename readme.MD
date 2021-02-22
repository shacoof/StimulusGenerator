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
fastSpeed/slowSpeed = step per milisecond, each step the shape is moved (screen_width/1000)/speed. Note : if screen_width < 1000 then movement will happen once every few miliseconds, e.g. 
    screen_width = 250 and speed = 1 means that every 4 milisecond the shape will move 1 pixel
    screen_widht = 250 and speed = 2 means that every 8 miliseconds the shape will move 1 pixel
    speed = 1 the fastest
fastDuration/slowDuration = how many cycles of fast/speed before switching to the other one
startMode = can be AFTER or WITH. a stimulus starts AFTER the previous one completed or WITH the previous one. in both cases delay (ms) can be applied
delay = see startMode


  ActionItem add reload of csv DONE
  ActionItem show/hide cross DONE
  ActionItem changing speed Done
  ActionItem subsequent stimuli are running with delay>=0 with previous one , like power-point DONE
  ActionItem increase size linearly 
  ActionItem add a label at the top with current stimulus info