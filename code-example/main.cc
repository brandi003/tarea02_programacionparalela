#include <global.hh>
#include <algorithm>
#include <RandomUnifStream.hpp>
#include <Timing.hpp>
#include <MatrixToMem.hpp>
#include <emmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>

/////////////////////////////////////////////////////////////////////////////////
//   Usage:
//           ./program_name  .......
//
//   Description:
//                ...................
//
/////////////////////////////////////////////////////////////////////////////////

void uso(std::string pname)
{
	std::cerr << "Uso: " << pname << " --fname MATRIX_FILE" << std::endl;
	exit(EXIT_FAILURE);
}

int main(int argc, char** argv)
{

	std::string fileName;
	
	//////////////////////////////////////////
	//  Read command-line parameters easy way
	if(argc != 3){
		uso(argv[0]);
	}
	std::string mystr;
	for (size_t i=0; i < argc; i++) {
		mystr=argv[i];
		if (mystr == "--fname") {
			fileName = argv[i+1];
		}
	}
	
	Timing timer0, timer1;
	////////////////////////////////////////////////////////////////
	// Transferir la matriz del archivo fileName a memoria principal
	timer0.start();
	MatrixToMem m1(fileName);
	timer0.stop();
	
	std::cout << "Time to transfer to main memory: " << timer0.elapsed() << std::endl;
	
	timer1.start();
	std::sort(m1._matrixInMemory, m1._matrixInMemory + m1._nfil);
	timer1.stop();
	
	std::cout << "Time to sort in main memory: " << timer1.elapsed() << std::endl;
	
	////////////////////////////////////////////////////////////////
	// Mostrar los 5 primeros elementos de la matriz ordenada.
	for(size_t i=0; i< 5; i++){		
		std::cout <<  m1._matrixInMemory[i] << std::endl;
	}

	
	
//////////////////////
	//aquiempieza mi basura
std::cout <<  m1._nfil << std::endl;
	//__m128i Registro1,Registro2;
	for (size_t i=0;i< m1._nfil;i+=8){
		//auto Registro1= _mm_loadl_epi32(&m1._matrixInMemory[i]);
		//auto Registro2= _mm_loadu_si32(&m1._matrixInMemory[i+4]);

		std::cout <<  loadu_epi32(&m1._matrixInMemory[i]) << std::endl;
		//std::cout <<  Registro1 << std::endl;
	}




	return(EXIT_SUCCESS);
}


