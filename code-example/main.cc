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


	uint32_t m1=_mm_extract_epi32(*Registro1,0);
	uint32_t m2=_mm_extract_epi32(*Registro1,1);
	uint32_t m3=_mm_extract_epi32(*Registro1,2);
	uint32_t m4=_mm_extract_epi32(*Registro1,3);
	uint32_t M1=_mm_extract_epi32(*Registro2,0);
	uint32_t M2=_mm_extract_epi32(*Registro2,1);
	uint32_t M3=_mm_extract_epi32(*Registro2,2);
	uint32_t M4=_mm_extract_epi32(*Registro2,3);
	*Registro1=_mm_setr_epi32(m1,M1,m2,M2);
	*Registro2=_mm_setr_epi32(m3,M3,m4,M4);
	aux=_mm_min_epi32(*Registro1,*Registro2);
	*Registro2=_mm_max_epi32(*Registro1,*Registro2);
	*Registro1=aux;


	m1=_mm_extract_epi32(*Registro1,0);
	m2=_mm_extract_epi32(*Registro1,1);
	m3=_mm_extract_epi32(*Registro1,2);
	m4=_mm_extract_epi32(*Registro1,3);
	M1=_mm_extract_epi32(*Registro2,0);
	M2=_mm_extract_epi32(*Registro2,1);
	M3=_mm_extract_epi32(*Registro2,2);
	M4=_mm_extract_epi32(*Registro2,3);
	*Registro1=_mm_setr_epi32(m1,M1,m2,M2);
	*Registro2=_mm_setr_epi32(m3,M3,m4,M4);
	aux=_mm_min_epi32(*Registro1,*Registro2);
	*Registro2=_mm_max_epi32(*Registro1,*Registro2);
	*Registro1=aux;


}

void bitonic_merge_network(__m128i* Registro1,__m128i* Registro2,__m128i* Registro3,__m128i* Registro4){
	bitonic_sorter(&*Registro1,&*Registro2);
	bitonic_sorter(&*Registro3,&*Registro4);

	bitonic_sorter(&*Registro2,&*Registro3);

	bitonic_sorter(&*Registro1,&*Registro2);
	bitonic_sorter(&*Registro3,&*Registro4);
}

void print_matriz(__m128i* Registros){
	std::cout << "-----------------Inicio de la matriz---------------------" << std::endl;
	std::cout << "[" << _mm_extract_epi32(Registros[0],0) << "," << _mm_extract_epi32(Registros[0],1) << "," << _mm_extract_epi32(Registros[0],2) << "," << _mm_extract_epi32(Registros[0],3) << "]" << std::endl;
	std::cout << "[" << _mm_extract_epi32(Registros[1],0) << "," << _mm_extract_epi32(Registros[1],1) << "," << _mm_extract_epi32(Registros[1],2) << "," << _mm_extract_epi32(Registros[1],3) << "]" << std::endl;
	std::cout << "[" << _mm_extract_epi32(Registros[2],0) << "," << _mm_extract_epi32(Registros[2],1) << "," << _mm_extract_epi32(Registros[2],2) << "," << _mm_extract_epi32(Registros[2],3) << "]" << std::endl;
	std::cout << "[" << _mm_extract_epi32(Registros[3],0) << "," << _mm_extract_epi32(Registros[3],1) << "," << _mm_extract_epi32(Registros[3],2) << "," << _mm_extract_epi32(Registros[3],3) << "]" << std::endl;
	std::cout << "-----------------Termino de la matriz---------------------" << std::endl;
}

int shellSort(MatrixToMem matriz, int n)
{
    // Start with a big gap, then reduce the gap
    for (int gap = n/2; gap > 0; gap /= 2)
    {
        // Do a gapped insertion sort for this gap size.
        // The first gap elements a[0..gap-1] are already in gapped order
        // keep adding one more element until the entire array is
        // gap sorted
        for (int i = gap; i < n; i += 1)
        {
            // add a[i] to the elements that have been gap sorted
            // save a[i] in temp and make a hole at position i
            int temp = matriz._matrixInMemory[i];
 
            // shift earlier gap-sorted elements up until the correct
            // location for a[i] is found
            int j;           
            for (j = i; j >= gap && matriz._matrixInMemory[j - gap] > temp; j -= gap)
                matriz._matrixInMemory[j] = matriz._matrixInMemory[j - gap];
             
            //  put temp (the original a[i]) in its correct location
            matriz._matrixInMemory[j] = temp;
        }
    }
    return 0;
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
		sorting_network(Registros);
		traspuesta(Registros);
		bitonic_merge_network(&Registros[0],&Registros[1],&Registros[2],&Registros[3]);
		traspuesta(Registros);
		m2._matrixInMemory[0]=_mm_extract_epi32(Registros[0],0);
		m2._matrixInMemory[1]=_mm_extract_epi32(Registros[0],1);
		m2._matrixInMemory[2]=_mm_extract_epi32(Registros[0],2);
		m2._matrixInMemory[3]=_mm_extract_epi32(Registros[0],3);
		m2._matrixInMemory[4]=_mm_extract_epi32(Registros[1],0);
		m2._matrixInMemory[5]=_mm_extract_epi32(Registros[1],1);
		m2._matrixInMemory[6]=_mm_extract_epi32(Registros[1],2);
		m2._matrixInMemory[7]=_mm_extract_epi32(Registros[1],3);
		m2._matrixInMemory[8]=_mm_extract_epi32(Registros[2],0);
		m2._matrixInMemory[9]=_mm_extract_epi32(Registros[2],1);
		m2._matrixInMemory[10]=_mm_extract_epi32(Registros[2],2);
		m2._matrixInMemory[11]=_mm_extract_epi32(Registros[2],3);
		m2._matrixInMemory[12]=_mm_extract_epi32(Registros[3],0);
		m2._matrixInMemory[13]=_mm_extract_epi32(Registros[3],1);
		m2._matrixInMemory[14]=_mm_extract_epi32(Registros[3],2);
		m2._matrixInMemory[15]=_mm_extract_epi32(Registros[3],3);
	}
	timer3.stop();
	shellSort(m2._matrixInMemory, m2._nfil);
	
	std::cout << "Time to sort in main memory: " << timer3.elapsed() << std::endl;
	
	////////////////////////////////////////////////////////////////
	// Mostrar los 5 primeros elementos de la matriz ordenada.
	for(size_t i=0; i< 5; i++){		
		std::cout <<  m2._matrixInMemory[i] << std::endl;
	}
	std::cout << "-------------------------------"<< std::endl;
	return(EXIT_SUCCESS);
}


