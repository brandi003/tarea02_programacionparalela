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
/*for (size_t i=0;i< m1._nfil;i+=2){
	Registro1= _mm_loadu_si64(&m1._matrixInMemory[1]);
	Registro2= _mm_loadu_si64(&m1._matrixInMemory[3]);

}*/
for (size_t i=0;i< m1._nfil;i+=2){
	Registro1= _mm_loadu_si64(&m1._matrixInMemory[i]);
	for (size_t j=0;i<m1._nfil;i+=2){
		Registro2= _mm_loadu_si64(&m1._matrixInMemory[j]);
		__m128i result =_mm_cmpgt_epi32(Registro1,Registro2);
		_mm_storeu_si64(vectorOut1,result);
		if(!vectorOut1[0]==0){
			std::cout << "swap" << std::endl;
		}
		if(!vectorOut1[1]==0){
			std::cout << "swap" << std::endl;
		}
	}
}
/*
for i in range(len(lista)):
	for j in range(len(lista)):
		aux=lista[j]
		if(lista[i]<lista[j]):
			lista[j]=lista[i]
			lista[i]=aux*/

//__m128i result =_mm_cmpgt_epi64(Registro1,Registro2);
/*
uint32_t *vectorOut1 = (uint32_t*)aligned_alloc (64, 8);
_mm_storeu_si64(vectorOut1,Registro1);
uint32_t *vectorOut2 = (uint32_t*)aligned_alloc (64, 8);
_mm_storeu_si64(vectorOut2,Registro2);
if(vectorOut1[0]<vectorOut2[0]){
	std::cout << "vectorOut1[0]<vectorOut2[0]" <<std::endl;
}else{
	std::cout << "vectorOut1[0]>vectorOut2[0]" <<std::endl;
}
if(vectorOut1[1]<vectorOut2[1]){
	std::cout << "vectorOut1[1]<vectorOut2[1]" <<std::endl;
}else{
	std::cout << "vectorOut1[1]>vectorOut2[1]" <<std::endl;
}

__m128i result =_mm_cmpgt_epi32(Registro1,Registro2);
_mm_storeu_si64(vectorOut1,result);
std::cout << "---------------" <<std::endl;
std::cout << vectorOut1[0] <<std::endl;
std::cout << vectorOut1[1] <<std::endl;*/
	return(EXIT_SUCCESS);
}


