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

void mostrar_registro(__m128i registro){
	uint32_t *vectorOut1 = (uint32_t*)aligned_alloc (64, 128);
	_mm_storeu_si64(vectorOut1,registro);
	std::cout << "---------------" <<std::endl;
	std::cout << vectorOut1[0] <<std::endl;
	std::cout << vectorOut1[1] <<std::endl;
	std::cout << vectorOut1[2] <<std::endl;
	std::cout << vectorOut1[3] <<std::endl;
	std::cout << "---------------" <<std::endl;

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


/*
std::cout << m1._nfil << std::endl;
for (size_t i=0;i< m1._nfil;i+=1){
	Registro1= _mm_set1_epi32 (m1._matrixInMemory[i]);
	for (size_t j=0;j<m1._nfil;j+=2){
		Registro2= _mm_loadu_si64(&m1._matrixInMemory[j]);
		__m128i result =_mm_cmpgt_epi32(Registro1,Registro2);
		uint32_t *vectorOut1 = (uint32_t*)aligned_alloc (64, 8);
		_mm_storeu_si64(vectorOut1,result);
		if(!(int)vectorOut1[0]==0){
			auto aux=m1._matrixInMemory[i];
			m1._matrixInMemory[i]=m1._matrixInMemory[j];
			m1._matrixInMemory[j]=aux;
			Registro1= _mm_set1_epi32 (m1._matrixInMemory[i]);
			Registro2= _mm_loadu_si64(&m1._matrixInMemory[j]);
		}
		if(!(int)vectorOut1[0]==0 && !(int)vectorOut1[1]==0){
			j=j-2;
			continue;
		}
		if(!(int)vectorOut1[1]==0){
			auto aux=m1._matrixInMemory[i];
			m1._matrixInMemory[i]=m1._matrixInMemory[j+1];
			m1._matrixInMemory[j+1]=aux;
			Registro1= _mm_set1_epi32 (m1._matrixInMemory[i]);
			Registro2= _mm_loadu_si64(&m1._matrixInMemory[j]);
		}
		
	}
}*/
for (size_t i=0;i< m1._nfil;i+=1){
	Registro1= _mm_set1_epi32 (m1._matrixInMemory[i]);
	for (size_t j=i;j<m1._nfil;j+=2){
		Registro2= _mm_loadu_si64(&m1._matrixInMemory[j]);
		__m128i result =_mm_sub_epi64(Registro1,Registro2);
		uint32_t *vectorOut1 = (uint32_t*)aligned_alloc (64, 8);
		_mm_storeu_si64(vectorOut1,result);
		if((int)vectorOut1[0]<=0 && (int)vectorOut1[1]<=0){
			continue;
		}else if((int)vectorOut1[0]>0 && (int)vectorOut1[1]<0){
			auto aux=m1._matrixInMemory[i];
			m1._matrixInMemory[i]=m1._matrixInMemory[j];
			m1._matrixInMemory[j]=aux;
			Registro1= _mm_set1_epi32 (m1._matrixInMemory[i]);
		}else if((int)vectorOut1[0]<0 && (int)vectorOut1[1]>0){
			auto aux=m1._matrixInMemory[i];
			m1._matrixInMemory[i]=m1._matrixInMemory[j+1];
			m1._matrixInMemory[j+1]=aux;
			Registro1= _mm_set1_epi32 (m1._matrixInMemory[i]);
		}else if((int)vectorOut1[0]>=(int)vectorOut1[1]){
			auto aux=m1._matrixInMemory[i];
			m1._matrixInMemory[i]=m1._matrixInMemory[j];
			m1._matrixInMemory[j]=aux;
			Registro1= _mm_set1_epi32 (m1._matrixInMemory[i]);
		}else if((int)vectorOut1[0]<(int)vectorOut1[1]){
			auto aux=m1._matrixInMemory[i];
			m1._matrixInMemory[i]=m1._matrixInMemory[j+1];
			m1._matrixInMemory[j+1]=aux;
			Registro1= _mm_set1_epi32 (m1._matrixInMemory[i]);
		}
		
	}
	
}
/*
uint32_t *vectorOut1 = (uint32_t*)aligned_alloc (8, 8);
Registro2= _mm_loadu_si64(&m1._matrixInMemory[999]);
_mm_storeu_si64(vectorOut1,Registro2);

std::cout <<  m1._matrixInMemory[999] << std::endl;
std::cout <<  vectorOut1[0] << std::endl;
std::cout <<  vectorOut1[1] << std::endl;*/


for(size_t i=0; i< m1._nfil; i++){		
		std::cout <<  m1._matrixInMemory[i] << std::endl;
	}
std::cout << "-------------------------------"<< std::endl;

	return(EXIT_SUCCESS);
}


