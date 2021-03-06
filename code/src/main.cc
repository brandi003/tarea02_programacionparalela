#include <global.hh>
#include <stdio.h>
#include <algorithm>
#include <RandomUnifStream.hpp>
#include <Timing.hpp>
#include <MatrixToMem.hpp>
#include <immintrin.h>
#include <emmintrin.h>
#include <iostream>

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

void sorting_network(__m128i* Registros){
	__m128i minp1=_mm_min_epi32(Registros[0],Registros[2]);
	__m128i maxp1=_mm_max_epi32(Registros[0],Registros[2]);
	__m128i minp2=_mm_min_epi32(Registros[1],Registros[3]);
	__m128i maxp2=_mm_max_epi32(Registros[1],Registros[3]);
	__m128i minp3=_mm_min_epi32(maxp1,maxp2);
	__m128i maxp3=_mm_max_epi32(maxp1,maxp2);
	__m128i minp4=_mm_min_epi32(minp1,minp2);
	__m128i maxp4=_mm_max_epi32(minp1,minp2);
	__m128i minp5=_mm_min_epi32(maxp4,minp3);
	__m128i maxp5=_mm_max_epi32(maxp4,minp3);
	Registros[0]=minp4;
	Registros[1]=minp5;
	Registros[2]=maxp5;
	Registros[3]=maxp3;

}



void traspuesta(__m128i* Registros){
	__m128i sub1 = _mm_unpackhi_epi32(Registros[0],Registros[1]);
	__m128i sub2 = _mm_unpackhi_epi32(Registros[2],Registros[3]);
	__m128i sub3 = _mm_unpacklo_epi32(Registros[0],Registros[1]);
	__m128i sub4 = _mm_unpacklo_epi32(Registros[2],Registros[3]);

	Registros[0]=_mm_unpacklo_epi64(sub3,sub4);
	Registros[1]=_mm_unpackhi_epi64(sub3,sub4);
	Registros[2]=_mm_unpacklo_epi64(sub1,sub2);
	Registros[3]=_mm_unpackhi_epi64(sub1,sub2);
}

void bitonic_sorter(__m128i* Registro1,__m128i* Registro2){
	*Registro2=_mm_shuffle_epi32(*Registro2, _MM_SHUFFLE(0, 1, 2, 3));
	auto aux=_mm_min_epi32(*Registro1,*Registro2);
	*Registro2=_mm_max_epi32(*Registro1,*Registro2);
	*Registro1=aux;

	auto sub1 = _mm_unpackhi_epi32(*Registro1,*Registro2);
	auto sub2 = _mm_unpacklo_epi32(*Registro1,*Registro2);
	*Registro1=_mm_min_epi32(sub1,sub2);
	*Registro2=_mm_max_epi32(sub1,sub2);

	sub1 = _mm_unpackhi_epi32(*Registro1,*Registro2);
	sub2 = _mm_unpacklo_epi32(*Registro1,*Registro2);
	*Registro1=_mm_min_epi32(sub1,sub2);
	*Registro2=_mm_max_epi32(sub1,sub2);
}

void bitonic_merge_network(__m128i* Registro1,__m128i* Registro2,__m128i* Registro3,__m128i* Registro4){
	bitonic_sorter(&*Registro1,&*Registro2);
	bitonic_sorter(&*Registro3,&*Registro4);

	bitonic_sorter(&*Registro2,&*Registro3);

	bitonic_sorter(&*Registro1,&*Registro2);
	bitonic_sorter(&*Registro3,&*Registro4);
}

void print_m2(__m128i* Registros){
	std::cout << "-----------------Inicio de la m2---------------------" << std::endl;
	std::cout << "[" << _mm_extract_epi32(Registros[0],0) << "," << _mm_extract_epi32(Registros[0],1) << "," << _mm_extract_epi32(Registros[0],2) << "," << _mm_extract_epi32(Registros[0],3) << "]" << std::endl;
	std::cout << "[" << _mm_extract_epi32(Registros[1],0) << "," << _mm_extract_epi32(Registros[1],1) << "," << _mm_extract_epi32(Registros[1],2) << "," << _mm_extract_epi32(Registros[1],3) << "]" << std::endl;
	std::cout << "[" << _mm_extract_epi32(Registros[2],0) << "," << _mm_extract_epi32(Registros[2],1) << "," << _mm_extract_epi32(Registros[2],2) << "," << _mm_extract_epi32(Registros[2],3) << "]" << std::endl;
	std::cout << "[" << _mm_extract_epi32(Registros[3],0) << "," << _mm_extract_epi32(Registros[3],1) << "," << _mm_extract_epi32(Registros[3],2) << "," << _mm_extract_epi32(Registros[3],3) << "]" << std::endl;
	std::cout << "-----------------Termino de la m2---------------------" << std::endl;
}

int main(int argc, char** argv)
{
	//timers y contadores
	double cont0=0, cont1=0, cont2=0, cont3=0, cont4=0;
	Timing timer0, timer1, timer2, timer3, timer4; 
	int repeticiones=1;
	for (size_t m=0;m<repeticiones;m++){
		std::cout << m << std::endl;
/******************************************************************************************************************************************************/
/******************************************************************************************************************************************************/
/******************************************************************************************************************************************************/
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
		////////////////////////////////////////////////////////////////
		// Transferir la m2 del archivo fileName a memoria principal
		timer0.start();
		MatrixToMem m1(fileName);
		timer0.stop();
		cont0=cont0+timer0.elapsed();
		//std::cout << "Time to transfer to main memory: " << timer0.elapsed() << std::endl;
		timer1.start();
		std::sort(m1._matrixInMemory, m1._matrixInMemory + m1._nfil);
		timer1.stop();
		cont1=cont1+timer1.elapsed();
		//std::cout << "Time to sort in main memory: " << timer1.elapsed() << std::endl;
		// Mostrar los 5 primeros elementos de la m2 ordenada.
		/*for(size_t i=0; i< 5; i++){		
			std::cout <<  m1._matrixInMemory[i] << std::endl;
		}
		std::cout << "-------------------------------"<< std::endl;*/
/******************************************************************************************************************************************************/
/******************************************************************************************************************************************************/
/******************************************************************************************************************************************************/



/******************************************************************************************************************************************************/
/******************************************************************************************************************************************************/
/******************************************************************************************************************************************************/
		MatrixToMem m2(fileName);
		//std::cout << "Time to transfer to main memory: " << timer2.elapsed() << std::endl;
		timer2.start();
		__m128i Registros[4];
		for (size_t i=0;i<m2._nfil;i+=16){
			if(m2._nfil==1000 && i==992){
				break;
			}
			Registros[0]=_mm_setr_epi32(m2._matrixInMemory[i],m2._matrixInMemory[i+1],m2._matrixInMemory[i+2],m2._matrixInMemory[i+3]);
			Registros[1]=_mm_setr_epi32(m2._matrixInMemory[i+4],m2._matrixInMemory[i+5],m2._matrixInMemory[i+6],m2._matrixInMemory[i+7]);
			Registros[2]=_mm_setr_epi32(m2._matrixInMemory[i+8],m2._matrixInMemory[i+9],m2._matrixInMemory[i+10],m2._matrixInMemory[i+11]);
			Registros[3]=_mm_setr_epi32(m2._matrixInMemory[i+12],m2._matrixInMemory[i+13],m2._matrixInMemory[i+14],m2._matrixInMemory[i+15]);
			sorting_network(Registros);
			traspuesta(Registros);
			bitonic_merge_network(&Registros[0],&Registros[1],&Registros[2],&Registros[3]);
			traspuesta(Registros);
			m2._matrixInMemory[i]=_mm_extract_epi32(Registros[0],0);
			m2._matrixInMemory[i+1]=_mm_extract_epi32(Registros[0],1);
			m2._matrixInMemory[i+2]=_mm_extract_epi32(Registros[0],2);
			m2._matrixInMemory[i+3]=_mm_extract_epi32(Registros[0],3);
			m2._matrixInMemory[i+4]=_mm_extract_epi32(Registros[1],0);
			m2._matrixInMemory[i+5]=_mm_extract_epi32(Registros[1],1);
			m2._matrixInMemory[i+6]=_mm_extract_epi32(Registros[1],2);
			m2._matrixInMemory[i+7]=_mm_extract_epi32(Registros[1],3);
			m2._matrixInMemory[i+8]=_mm_extract_epi32(Registros[2],0);
			m2._matrixInMemory[i+9]=_mm_extract_epi32(Registros[2],1);
			m2._matrixInMemory[i+10]=_mm_extract_epi32(Registros[2],2);
			m2._matrixInMemory[i+11]=_mm_extract_epi32(Registros[2],3);
			m2._matrixInMemory[i+12]=_mm_extract_epi32(Registros[3],0);
			m2._matrixInMemory[i+13]=_mm_extract_epi32(Registros[3],1);
			m2._matrixInMemory[i+14]=_mm_extract_epi32(Registros[3],2);
			m2._matrixInMemory[i+15]=_mm_extract_epi32(Registros[3],3);
		}
		timer2.stop();
		cont2=cont2+timer2.elapsed();
		timer3.start();
		std::sort(m2._matrixInMemory, m2._matrixInMemory + m2._nfil);
		timer3.stop();
		cont3=cont3+timer3.elapsed();

/******************************************************************************************************************************************************/
/******************************************************************************************************************************************************/
/******************************************************************************************************************************************************/




/******************************************************************************************************************************************************/
/******************************************************************************************************************************************************/
/******************************************************************************************************************************************************/
		MatrixToMem m3(fileName);
		for (size_t i=0;i<m3._nfil;i+=16){
			if(m3._nfil==1000 && i==992){
				break;
			}
			Registros[0]=_mm_setr_epi32(m3._matrixInMemory[i],m3._matrixInMemory[i+1],m3._matrixInMemory[i+2],m3._matrixInMemory[i+3]);
			Registros[1]=_mm_setr_epi32(m3._matrixInMemory[i+4],m3._matrixInMemory[i+5],m3._matrixInMemory[i+6],m3._matrixInMemory[i+7]);
			Registros[2]=_mm_setr_epi32(m3._matrixInMemory[i+8],m3._matrixInMemory[i+9],m3._matrixInMemory[i+10],m3._matrixInMemory[i+11]);
			Registros[3]=_mm_setr_epi32(m3._matrixInMemory[i+12],m3._matrixInMemory[i+13],m3._matrixInMemory[i+14],m3._matrixInMemory[i+15]);
			sorting_network(Registros);
			traspuesta(Registros);
			bitonic_merge_network(&Registros[0],&Registros[1],&Registros[2],&Registros[3]);
			traspuesta(Registros);
			m3._matrixInMemory[i]=_mm_extract_epi32(Registros[0],0);
			m3._matrixInMemory[i+1]=_mm_extract_epi32(Registros[0],1);
			m3._matrixInMemory[i+2]=_mm_extract_epi32(Registros[0],2);
			m3._matrixInMemory[i+3]=_mm_extract_epi32(Registros[0],3);
			m3._matrixInMemory[i+4]=_mm_extract_epi32(Registros[1],0);
			m3._matrixInMemory[i+5]=_mm_extract_epi32(Registros[1],1);
			m3._matrixInMemory[i+6]=_mm_extract_epi32(Registros[1],2);
			m3._matrixInMemory[i+7]=_mm_extract_epi32(Registros[1],3);
			m3._matrixInMemory[i+8]=_mm_extract_epi32(Registros[2],0);
			m3._matrixInMemory[i+9]=_mm_extract_epi32(Registros[2],1);
			m3._matrixInMemory[i+10]=_mm_extract_epi32(Registros[2],2);
			m3._matrixInMemory[i+11]=_mm_extract_epi32(Registros[2],3);
			m3._matrixInMemory[i+12]=_mm_extract_epi32(Registros[3],0);
			m3._matrixInMemory[i+13]=_mm_extract_epi32(Registros[3],1);
			m3._matrixInMemory[i+14]=_mm_extract_epi32(Registros[3],2);
			m3._matrixInMemory[i+15]=_mm_extract_epi32(Registros[3],3);
		}
		timer4.start();
		for (uint32_t gap = m3._nfil/2; gap > 0; gap /= 2)
	    {
	        for (uint32_t i = gap; i < m3._nfil; i += 1)
	        {
	            uint32_t temp = m3._matrixInMemory[i];
	            uint32_t j;           
	            for (j = i; j >= gap && m3._matrixInMemory[j - gap] > temp; j -= gap)
	                m3._matrixInMemory[j] = m3._matrixInMemory[j - gap];
	            m3._matrixInMemory[j] = temp;
	        }
	    }
	    timer4.stop();
	    cont4=cont4+timer4.elapsed();

		/*
		std::cout << "Time to sort in main memory: " << timer3.elapsed() << std::endl;
		////////////////////////////////////////////////////////////////
		// Mostrar los 5 primeros elementos de la m2 ordenada.
		for(size_t i=0; i< 10; i++){		
			std::cout <<  m2._matrixInMemory[i] << std::endl;
		}
		std::cout << "-------------------------------"<< std::endl;*/
	}
	std::cout << "------------------------------------------------------------"<< std::endl;
	std::cout << "tiempo de carga de la matriz a memoria " << (cont0/repeticiones) << std::endl;
	std::cout << "tiempo de ordenamiento con std::sort " << (cont1/repeticiones) << std::endl;
	std::cout << "tiempo de preordenamiento con procesamiento vectorial " << (cont2/repeticiones) << std::endl;
	std::cout << "tiempo de ordenamiento con std::sort de la matriz preordenada " << (cont3/repeticiones) << std::endl;
	std::cout << "tiempo de ordenamiento con procesamiento vectorial " << (cont2+cont3)/(repeticiones) << std::endl;
	std::cout << "tiempo de ordenamiento con shellSort de la matriz preordenada " << (cont4+cont3/repeticiones) << std::endl;
	std::cout << "------------------------------------------------------------"<< std::endl;
	return(EXIT_SUCCESS);
}


