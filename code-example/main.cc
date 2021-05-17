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
	__m128i Registros[4];
	for (size_t i=0;i<m2._nfil;i+=16){
		Registros[0]=_mm_setr_epi32(m2._matrixInMemory[i],m2._matrixInMemory[i+1],m2._matrixInMemory[i+2],m2._matrixInMemory[i+3]);
		Registros[1]=_mm_setr_epi32(m2._matrixInMemory[i+4],m2._matrixInMemory[i+5],m2._matrixInMemory[i+6],m2._matrixInMemory[i+7]);
		Registros[2]=_mm_setr_epi32(m2._matrixInMemory[i+8],m2._matrixInMemory[i+9],m2._matrixInMemory[i+10],m2._matrixInMemory[i+11]);
		Registros[3]=_mm_setr_epi32(m2._matrixInMemory[i+12],m2._matrixInMemory[i+13],m2._matrixInMemory[i+14],m2._matrixInMemory[i+15]);
		std::cout << _mm_extract_epi32(Registros[0],0) << std::endl;
		std::cout << _mm_extract_epi32(Registros[0],1) << std::endl;
		std::cout << _mm_extract_epi32(Registros[0],2) << std::endl;
		std::cout << _mm_extract_epi32(Registros[0],3) << std::endl;
		std::cout << _mm_extract_epi32(Registros[1],0) << std::endl;
		std::cout << _mm_extract_epi32(Registros[1],1) << std::endl;
		std::cout << _mm_extract_epi32(Registros[1],2) << std::endl;
		std::cout << _mm_extract_epi32(Registros[1],3) << std::endl;
		std::cout << _mm_extract_epi32(Registros[2],0) << std::endl;
		std::cout << _mm_extract_epi32(Registros[2],1) << std::endl;
		std::cout << _mm_extract_epi32(Registros[2],2) << std::endl;
		std::cout << _mm_extract_epi32(Registros[2],3) << std::endl;
		std::cout << _mm_extract_epi32(Registros[3],0) << std::endl;
		std::cout << _mm_extract_epi32(Registros[3],1) << std::endl;
		std::cout << _mm_extract_epi32(Registros[3],2) << std::endl;
		std::cout << _mm_extract_epi32(Registros[3],3) << std::endl;
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


