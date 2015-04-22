

#include "Class_VTK.hpp"

using namespace std;


// =================================================================================== //
void VTK::Write_Data_Header( fstream &str, bool parallel ){

  string        location ;
  stringstream  scalars, vectors ;


  for( int j=0; j<2; j++){

    if( j==0 ) location = "Point" ;
    if( j==1 ) location = "Cell" ;

//    scalars.clear();
//    vectors.clear();

    scalars.str("");
    vectors.str("");

    //Creating Scalar and Vector Lists
    scalars << "\"" ;
    vectors << "\"" ;

    for( int i=0; i< nr_data; i++ ){

      if( data[i].Get_Location() == location){
        if(      data[i].Get_Components() == 1 ) scalars <<  data[i].Get_Name() << " " ;
        else if( data[i].Get_Components() == 3 ) vectors <<  data[i].Get_Name() << " " ;
      };

    };

    scalars << "\"" ;
    vectors << "\"" ;

    if(      location == "Point") {
      str << "      <" ;
      if( parallel )  str << "P" ;
      str << "PointData " ;
    }

    else if( location == "Cell")  {
      str << "      <" ;
      if( parallel )  str << "P" ;
      str << "CellData " ;
    };

    str << " Scalars=" << scalars.str()
        << " Vectors=" << vectors.str()
        << ">" << endl;

    //Writing DataArray
    for( int i=0; i< nr_data; i++){
      if( data[i].Get_Location() == location && !parallel) Write_DataArray( str, data[i] ) ;
      if( data[i].Get_Location() == location &&  parallel) Write_PDataArray( str, data[i] ); 
    };

    str << "      </" ;
    if( parallel )  str << "P" ;

    if( location == "Point") str << "PointData> " << endl;
    if( location == "Cell")  str << "CellData> "  << endl;

  };

  return ;

};

// =================================================================================== //
void  VTK::Write_DataArray( fstream &str, VTK::Field_C &field_ ){

  str << "        <DataArray "
       << "type=\"" << field_.Get_Type() << "\" "
       << "Name=\"" << field_.Get_Name() << "\" "
       << "NumberOfComponents=\""<< field_.Get_Components() << "\" "
       << "format=\"" << field_.Get_Codification() << "\" ";

  if( field_.Get_Codification() == "appended"){
    str << "offset=\"" << field_.Get_Offset() << "\" " ;
  };
  
  str << ">" ;

  if( field_.Get_Codification() == "ascii") {
    str << endl;
    Flush( str, "ascii", field_.Get_Name() ) ;
    str << "       " << endl;
  };

  str << " </DataArray>" << endl;
       
       
  return ;
  
};

// =================================================================================== //
void  VTK::Write_PDataArray( fstream &str, VTK::Field_C &field_ ){

  str << "        <PDataArray "
      << "type=\"" << field_.Get_Type() << "\" "
      << "Name=\"" << field_.Get_Name() << "\" "
      << "NumberOfComponents=\""<< field_.Get_Components() << "\" " 
      << "/>" ;

  str << endl ;

  return ;
  
};

// =================================================================================== //

void VTK::Write_All_Appended( fstream &str ){

  int nbytes ;

  //Start appended section
  str << "  <AppendedData encoding=\"raw\">" << endl;
  str << "_" ;
  str.close();
  str.clear();

  //Reopening in binary mode
  str.open( fh.Get_Name( ), ios::out |ios::app| ios::binary);

  //Writing first point data then cell data
  for( int i=0; i< nr_data; i++){
    if( data[i].Get_Codification() == "appended" && data[i].Get_Location() == "Point") {
      nbytes = data[i].Get_Nbytes() ;
      flush_binary( str, nbytes  ) ;
      Flush( str, "binary", data[i].Get_Name() ) ;
    };
  } 
  

  for( int i=0; i< nr_data; i++){
    if( data[i].Get_Codification() == "appended" && data[i].Get_Location() == "Cell") {
      nbytes = data[i].Get_Nbytes()  ;
      str.write( reinterpret_cast<char*>(&nbytes), sizeof (int) ) ;
      Flush( str, "binary", data[i].Get_Name() ) ;
    };
  } 

  //Writing Geometry Data
  for(int i=0; i<geometry.size(); i++){
    if( geometry[i].Get_Codification() == "appended" ) {
      nbytes = geometry[i].Get_Nbytes()  ;
      str.write( reinterpret_cast<char*>(&nbytes), sizeof (int) ) ;
      Flush( str, "binary", geometry[i].Get_Name() ) ;           
    };
  };

  // Closing Appended Secyion
  str.close();
  str.clear();

  str.open( fh.Get_Name( ), ios::out |ios::app) ;
  str << endl;
  str << "  </AppendedData>" << endl;

  return ;

};
