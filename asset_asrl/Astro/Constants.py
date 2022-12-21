pi   = 3.14159265358979;
Gcon = 6.67259e-11;
AU   = 149597870700.0;
LD   = 3.84402e8
meter = 1.0;
kilometer = 1000.0;
sec = 1.0;
minute = 60.0;
hour = 3600.0;
day = 24.0*3600.0;
year = 365.0*day;
dtr = pi/180.0;
rtd = 180/pi;




MuSun   = 1.3271244004193938e20;


MuEarth = 3.9860043543609598e14;
MuMoon = 4.9028000661637961e12;
MuMars = 4.2828372e13;
MuJupiter = 1.26686534921801e17;
MuJupiterBarycenter = 1.267127648e17
MuSaturn =3.79312074986522e16
MuSaturnBarycenter =3.79405852000000e16

MuVenus = 3.24858592e14;
MuMercury = 2.203178e13;
MuUranus =5.79395132227901e15
MuUranusBarycenter = 5.79454860000001e15
MuNeptune = 6.83652710058002e15
MuNeptuneBarycenter = 6.83652710058002e15



JupiterMass = MuJupiter/Gcon 
SaturnMass = MuSaturn/Gcon 
VenusMass = MuVenus/Gcon 
MarsMass = MuMars/Gcon 

MercuryMass = MuMercury/Gcon 
UranusMass  = MuUranus/Gcon 
NeptuneMass = MuNeptune/Gcon 

EarthMass = MuEarth/Gcon
MoonMass = MuMoon/Gcon
SunMass =MuSun/Gcon


RadiusSun       = 696000.0*1000.0
RadiusMercury   = 2440.5*1000.0
RadiusEarth     = 6378.136*1000.0
RadiusMoon      = 1737.4*1000.0
RadiusVenus     = 6052.0*1000.0
RadiusMars      = 3397.2*1000.0
RadiusJupiter   = 71398*1000.0
RadiusSaturn    = 60000*1000.0
RadiusUranus    = 25400*1000.0
RadiusNeptune   = 24300*1000.0


J2Earth  = .001082629
J2Mars   = .001955
J2Venus  = 4.4044e-6
J2Moon   = 2.0323e-4



SpiceBodyProps={}
SpiceBodyProps["SUN"]  ={"Mu":MuSun,"Radius":RadiusSun}
SpiceBodyProps["EARTH"]={"Mu":MuEarth,"J2":J2Earth,"Radius":RadiusEarth}
SpiceBodyProps["MOON"] ={"Mu":MuMoon,"Radius":RadiusMoon}
SpiceBodyProps["MERCURY"] ={"Mu":MuMercury,"Radius":RadiusMercury}
SpiceBodyProps["VENUS"]={"Mu":MuVenus,"Radius":RadiusVenus}
SpiceBodyProps["JUPITER BARYCENTER"]={"Mu":MuJupiterBarycenter,"Radius":RadiusJupiter}
SpiceBodyProps["SATURN BARYCENTER"] ={"Mu":MuSaturnBarycenter}
SpiceBodyProps["NEPTUNE BARYCENTER"]={"Mu":MuNeptune}
SpiceBodyProps["URANUS BARYCENTER"] ={"Mu":MuUranus}
SpiceBodyProps["MARS BARYCENTER"]={"Mu":MuMars}

