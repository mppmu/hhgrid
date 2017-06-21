      logical function equal(a, b)
      implicit none
      real*8, intent(in) :: a, b
      if ( abs(a-b) > 1.0D-14 ) then
        equal = .false.
      else
        equal = .true.
      endif
      end function

      program testgrid
      use, intrinsic :: iso_c_binding
      implicit none
      real(c_double) :: s, t, result, expected
      type(c_ptr) :: grid
      integer :: i

      interface
        subroutine python_initialize() bind(c)
          use, intrinsic :: iso_c_binding
          implicit none
        end subroutine python_initialize

        subroutine python_decref(grid) bind(c)
          use, intrinsic :: iso_c_binding
          implicit none
          type(c_ptr), intent(in) :: grid
        end subroutine python_decref

        subroutine python_finalize() bind(c)
          use, intrinsic :: iso_c_binding
          implicit none
        end subroutine python_finalize

        subroutine python_printinfo() bind(c)
          use, intrinsic :: iso_c_binding
          implicit none
        end subroutine python_printinfo

        type(c_ptr) function grid_initialize(grid_name) bind(c)
          use, intrinsic :: iso_c_binding
          implicit none
          character(kind=c_char) :: grid_name(*)
        end function grid_initialize

        real(c_double) function grid_virt(grid, s, t) bind(c)
          use, intrinsic :: iso_c_binding
          implicit none
          type(c_ptr), intent(in), value :: grid
          real(c_double), intent(in), value :: s, t
        end function grid_virt
      end interface

      call python_initialize
      call python_printinfo
      grid = grid_initialize(C_CHAR_"Virt_EFT.grid"//C_NULL_CHAR)

      ! Test point 1
      s = 250000.D0
      t = -50000.D0
      expected = 3.3875286683507290375755305333882461127359D-04
      if(.not. test_point(grid,s,t,expected)) then
         call EXIT(1)
      endif

      ! Test point 2
      s = 66887.6D0
      t = -9407.43D0
      expected = 7.6844534557689569490555349384752759078765D-07
      if(.not. test_point(grid,s,t,expected)) then
         call EXIT(1)
      endif

      ! Test point 3
      s = 83055.9D0
      t =  -11438.D0
      expected = 5.6368756265788111468190146879919666389469D-06
      if(.not. test_point(grid,s,t,expected)) then
         call EXIT(1)
      endif

      ! Test point 4
      s = 414983.D0
      t = -59786.9D0
      expected =  1.0669994178547940674728344845334504498169D-03
      if(.not. test_point(grid,s,t,expected)) then
         call EXIT(1)
      endif

      ! Test point 5
      s = 2.56513D6
      t = -482321.D0
      expected = 3.9170655780375776555679578905255766585469D-02
      if(.not. test_point(grid,s,t,expected)) then
         call EXIT(1)
      endif

      ! Evaluate point repeatedly (can detect synchronization issues with the grid)
      do i = 1, 10000
         if(.not. test_point(grid,s,t,expected)) then
            call EXIT(1)
         endif
      end do

      ! Test point that previously landed outside grid due to numerics, gave 0 (for Virt_EFT)
      s = 275621.46328957588411867618560791015625D0;
      t = -243368.2897650574450381100177764892578125D0;
      expected = 4.3499409718481744750728790194216344389133D-04;
      if(.not. test_point(grid,s,t,expected)) then
         call EXIT(1)
      endif

      ! Test point that previously landed outside grid due to numerics, gave 0 (for Virt_EFT)
      s = 959959.10608519916422665119171142578125D0;
      t = -928446.149935275432653725147247314453125D0;
      expected = 5.9433502098979523395327895229911518981680D-03;
      if(.not. test_point(grid,s,t,expected)) then
         call EXIT(1)
      endif

      call python_decref(grid)
      call python_finalize

      write(*,*) ' '
      write(*,*) 'TESTS PASSED'
      write(*,*) ' '

      contains

        logical function test_point(grid,s,t,expected)
        use, intrinsic :: iso_c_binding
        implicit none
        type(c_ptr), intent(in), value :: grid
        real*8, intent(in), value :: s, t, expected
        logical :: equal
        real*8 :: res

        write(*,*) 'Sending: ', s,' ', t
        res = grid_virt(grid, s, t)
        write(*,*) 'Received: ', res

        if (.not. equal(res,expected)) then
          write(*,*) 'Expected: ', expected, ', Got: ', res
          write(*,*) 'TESTS FAILED'

          ! Destruct grid, terminate python
          call python_decref(grid)
          call python_finalize

          test_point = .false.
        else
          test_point = .true.
        endif
        end function

      end program
