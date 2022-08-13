import MKgSecConstants as c

SpiceBodyProps={}


SpiceBodyProps["SUN"]  ={"Mu":c.MuSun,"Radius":c.RadiusSun}
SpiceBodyProps["EARTH"]={"Mu":c.MuEarth,"J2":c.J2Earth,"Radius":c.RadiusEarth}
SpiceBodyProps["MOON"] ={"Mu":c.MuMoon,"Radius":c.RadiusMoon}
SpiceBodyProps["MERCURY"] ={"Mu":c.MuMercury,"Radius":c.RadiusMercury}

SpiceBodyProps["VENUS"]={"Mu":c.MuVenus,"Radius":c.RadiusVenus}
SpiceBodyProps["JUPITER BARYCENTER"]={"Mu":c.MuJupiterBarycenter,"Radius":c.RadiusJupiter}
SpiceBodyProps["SATURN BARYCENTER"] ={"Mu":c.MuSaturnBarycenter}
SpiceBodyProps["NEPTUNE BARYCENTER"]={"Mu":c.MuNeptune}
SpiceBodyProps["URANUS BARYCENTER"] ={"Mu":c.MuUranus}
SpiceBodyProps["MARS BARYCENTER"]={"Mu":c.MuMars}
