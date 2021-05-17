#include <global.hh>
#include <stdio.h>
#include <algorithm>
#include <RandomUnifStream.hpp>
#include <Timing.hpp>
#include <MatrixToMem.hpp>
#include <immintrin.h>
#include <emmintrin.h>


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
	std::cout << "-------------------------------"<< std::endl;
/////////////////////


//////////////////////////////////////////////////////////////////////////////
	///////////////ejecucion de bubble sort con intrinsecas///////////

	Timing timer2, timer3;
	////////////////////////////////////////////////////////////////
	// Transferir la matriz del archivo fileName a memoria principal
	timer2.start();
	MatrixToMem m2(fileName);
	timer2.stop();
	std::cout << "Time to transfer to main memory: " << timer2.elapsed() << std::endl;
	timer3.start();
	__m128i Registro1,Registro2,Registro3,Registro4;
	for (size_t i=0;i<m2._nfil;i+=16){
		Registro1=_mm_loadu_si32(&m2._matrixInMemory[i]);
		Registro2=_mm_loadu_si32(&m2._matrixInMemory[i+4]);
		Registro3=_mm_loadu_si32(&m2._matrixInMemory[i+8]);
		Registro4=_mm_loadu_si32(&m2._matrixInMemory[i+12]);
		std::cout << _mm_extract_epi32(Registro1,0) << std::endl;
		std::cout << _mm_extract_epi32(Registro1,1) << std::endl;
		std::cout << _mm_extract_epi32(Registro1,2) << std::endl;
		std::cout << _mm_extract_epi32(Registro1,3) << std::endl;
		std::cout << _mm_extract_epi32(Registro2,0) << std::endl;
		std::cout << _mm_extract_epi32(Registro2,1) << std::endl;
		std::cout << _mm_extract_epi32(Registro2,2) << std::endl;
		std::cout << _mm_extract_epi32(Registro2,3) << std::endl;
		std::cout << _mm_extract_epi32(Registro3,0) << std::endl;
		std::cout << _mm_extract_epi32(Registro3,1) << std::endl;
		std::cout << _mm_extract_epi32(Registro3,2) << std::endl;
		std::cout << _mm_extract_epi32(Registro3,3) << std::endl;
		std::cout << _mm_extract_epi32(Registro4,0) << std::endl;
		std::cout << _mm_extract_epi32(Registro4,1) << std::endl;
		std::cout << _mm_extract_epi32(Registro4,2) << std::endl;
		std::cout << _mm_extract_epi32(Registro4,3) << std::endl;
		for (size_t i=0;i<16;i++){
			std::cout << m2._matrixInMemory[i] << std::endl;
		}






		break;
	}
	timer3.stop();
	
	std::cout << "Time to sort in main memory: " << timer3.elapsed() << std::endl;
	
	////////////////////////////////////////////////////////////////
	// Mostrar los 5 primeros elementos de la matriz ordenada.
	for(size_t i=0; i< 5; i++){		
		std::cout <<  m1._matrixInMemory[i] << std::endl;
	}
	std::cout << "-------------------------------"<< std::endl;
	return(EXIT_SUCCESS);
}


