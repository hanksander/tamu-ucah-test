# path = f'C:\Users\j-min\Downloads'
# pd.read_csv(f'{path}\paramaterizations.csv')

parameter_name = 'wingspan'         # what it's called in Python 
Parent_body_name = 'Strakes'        # the name of the body it is attached to in the Geom Browser tree
Name = 'TotalSpan'                  # Name in openvsp
Group = 'WingGeom'                  # Group in openvsp

initia = f"""
        self._{parameter_name}_id = get_id("{Parent_body_name}", "{Group}", "{Name}")
        self._{parameter_name} = vsp.GetParmVal(self._{parameter_name}_id)"""

getter = f"""
    @property
    def {parameter_name}(self):
        self.airframe_mesh = None # destroy the oudated mesh
        return self._{parameter_name}"""

setter = f"""
    @{parameter_name}.setter
    def length(self, value):
        self._{parameter_name} = value
        self.vsp.SetParmVal(self._{parameter_name}_id, value)"""


print(f'initialization:\n\n{initia}\n\n\n'
      f'getter & setter:\n\n{getter}\n'
      f'{setter}\n\n\n')

