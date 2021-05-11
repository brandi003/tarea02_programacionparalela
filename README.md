# Ordenar arreglo de enteros con SIMD

El objetivo de la tarea es introducir el uso de las instrucciones *intrinsics* de los procesadores Intel. Para esto, se ordenará un arreglo dinámico de enteros de 32 bits a través de operaciones vectoriales y se comparará el desempeño temporal del ordenamiento logrado con la función ```std::sort()```, la que tiene complejidad O(n·log n). Utilizará el concepto de *SpeedUp* de un programa para medir el desempeño temporal y los resultados de los experimentos serán mostrados a través de gráficos con su correspondiente explicación y análisis.

# Descripción del trabajo a realizar

## Trabajo experimental

El trabajo a realizar se dividirá principalmente en las siguientes etapas. Los pasos 1 y 3 están implementados en el código de ejemplo.

1. Cargar en memoria principal el contenido de un archivo que contiene los números a ordenar. El archivo contiene 10 millones de números enteros sin signo de 32 bits.
2. Medir el tiempo de carga en milisegundos.

Las pasos 3 al 6 se deben repetir para N={10<sup>3</sup>,10<sup>4</sup>,10<sup>5</sup>,10<sup>6</sup>, 10<sup>7</sup>}

3. Utilizar la función ```std::sort()``` para ordenar de menor a mayor en memoria principal el arreglo dinámico del punto 1), tomando los N primeros números.
4. Medir el tiempo de ordamiento del algoritmo del punto 3).
4. Implementar con instrinsics un algoritmo SIMD de ordenamiento de menor a mayor para la misma cantidad N números.
5. Medir el tiempo de ordenamiento del algoritmo SIMD del punto 4).
6. Mostrar el SpeedUp alcanzado.

El código de su trabajo deberá ser copiado en el directorio ```/projects/$USER/tarea02``` para su revisión. La estructura de este directorio debe ser:

```
/projects/$USER/
	+ tarea02/
		+ code/
			- Makefile
			+ src/
		- README.txt	
```

El archivo README.txt deberá contener una descripción de los directorios y de su contenido. El directorio ```src``` debe tener la misma estructura del que se encuentra en este repositorio. Recuerde que su código debe ser compilado con el comando ```make```. Una vez compilado sin errores, se ejecutará con la secuencia:

```
$ cd /projects/$USER/tarea02/code
$ make
$ ./tarea02 --fname /projects/tarea02/data_to_sort_10000000.dat	
Salida de los experimentos realizados...
```


## Informe escrito

Usted debe preparar un informe del trabajo realizado, utilizando la plantilla entregada ```plantilla-informe.dotx```. 
