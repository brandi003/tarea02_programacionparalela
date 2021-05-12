#include <global.hh>
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
	//std::sort(m1._matrixInMemory, m1._matrixInMemory + m1._nfil);
	timer1.stop();
	
	std::cout << "Time to sort in main memory: " << timer1.elapsed() << std::endl;
	
	////////////////////////////////////////////////////////////////
	// Mostrar los 5 primeros elementos de la matriz ordenada.
	for(size_t i=0; i< 5; i++){		
		std::cout <<  m1._matrixInMemory[i] << std::endl;
	}
	std::cout << "-------------------------------"<< std::endl;

	
	
//////////////////////
	//aquiempieza mi basura
std::cout << "tamaño arreglo:  " << m1._nfil << std::endl;
__m128i Registro1,Registro2;
//for (size_t i=0;i< m1._nfil;i+=2){
	Registro1= _mm_loadu_si64(&m1._matrixInMemory[0]);
	//Registro2= _mm_loadu_si64(&m1._matrixInMemory[0]);

//}

Registro1 =_mm_move_epi64(Registro1);
uint32_t *vectorOut1 = (uint32_t*)aligned_alloc (32, 8);
_mm_storeu_si64(vectorOut1,Registro1);
std::cout <<  vectorOut1[0] << std::endl;
std::cout <<  vectorOut1[1] << std::endl;

	return(EXIT_SUCCESS);
}


