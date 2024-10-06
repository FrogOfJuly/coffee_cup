import math
import scipy
import numpy as np
import matplotlib.pyplot as plt

# This is a class describing the coffee cup problem
class CoffeeCup:
    @staticmethod
    def q(t): 
        return 1 - (t - 1.0)*(t - 1.0)

    @staticmethod
    def p(t):
        return math.asin(t - 1.0)

    @staticmethod
    def SF(t):
        return math.pi / 2.0 + CoffeeCup.p(t) + (t - 1.0) * math.pow(CoffeeCup.q(t), 0.5)
    
    @staticmethod
    def VF(t): 
        return math.pi/ 2.0 * (t - 1) + math.pow(CoffeeCup.q(t), 0.5) - 1.0/3.0 * math.pow(CoffeeCup.q(t), 1.5) + (t - 1.0)*CoffeeCup.p(t)

    def __init__(self, a, D0):
        self.a = a
        self.D0 = D0

    # \tan{\alpha} angle
    def tan_alpha(self, t):
        return self.a/CoffeeCup.VF(t)

    # \alpha angle
    def alpha(self, t):
        return math.atan(self.tan_alpha(t))

    # Spill distance
    def D(self, t):
        return t * self.tan_alpha(t)

    # Area
    def A(self, t):
        tg = self.tan_alpha(t)
        return math.pow(1.0 + tg*tg, 0.5) * CoffeeCup.SF(t)
    
    # Touches base
    @property
    def alpha0(self):
        return math.atan(self.a/math.pi)

    # In the middle of base
    @property
    def alpha1(self):
        return math.atan(3.0*self.a/2.0)
        
    # ===== Numerical things =====

    ## Reversing alpha

    def tan_alpha_reverse(self, tan_alpha):
        # tan_alpha -> t
        # As the CoffeeCup.tan_alpha function is monotone on t \in (0, 2] it is possible to numerically reverse it via binary search
        # t \in (0,2] corresponds to \alpha being in (\pi/2, \alpha_0 or \alpha_2)

        left = math.pow(10.0, -6.0)
        right = 2.0

        def obj_fun(x):
            return self.tan_alpha(x) - tan_alpha

        root, r = scipy.optimize.brentq(obj_fun, left, right, full_output=True)

        if r.converged:
            return root
        else:
            print(r)
            return None
        
    ## Finding derivative zeros

    @property
    def A_extremes(self):
        # It's not A'(t), but an equivalent thing such that A'(t) = 0 \iff A_prime_eff(t) = a^2
        def A_prime_eff(t):
            sf = CoffeeCup.SF(t)
            vf = CoffeeCup.VF(t)
            q  = math.pow(CoffeeCup.q(t), 0.5)
            up = 2*q*vf*vf*vf
            down = sf*sf - 2*q*vf
            if up == 0.0 and down == 0.0:
                return 0.0
            else:
                return up/down
            
        # We need to locate roots to employ numeric solver. 
        # From the plot of A_prime_eff it is clear that if there are two roots of 
        # A_prime_eff(t) = a^2, then they are on different sides of its maximum.
            
        r = scipy.optimize.minimize_scalar( lambda x : -A_prime_eff(x)
                                      , bounds=[0.0, 2.0]
                                      , bracket=[0.0, 1.0, 2.0]
                                      , method="bounded")
        if not r.success:
            print(f"Failed to find extreme of A'eff: {r.message}")
            return None

        middle, a_max_sq = r.x, A_prime_eff(r.x)

        if self.a*self.a > a_max_sq:
            return []
        elif self.a*self.a == a_max_sq:
            return [(math.atan(self.tan_alpha(middle)), self.A(middle))]
        
        # Provided there is an extreme and it's value is greater than current a^2,
        # there are two roots on each side from middle

        def obj_fun(x):
            return A_prime_eff(x) - self.a*self.a


        left_root, left_result = scipy.optimize.brentq(obj_fun, 0.0, middle, full_output=True)

        if not left_result.converged:
            print(f"left root failed to converge: {left_result.flag}")
            return None

        right_root, right_result = scipy.optimize.brentq(obj_fun, middle, 2.0, full_output=True)

        if not right_result.converged:
            print(f"right root failed to converge: {right_result.flag}")
            return None

        return [ (math.atan(self.tan_alpha(left_root)) , self.A(left_root))
               , (math.atan(self.tan_alpha(right_root)), self.A(right_root))
        ]


# Class for drawing everything interactively

class CoffeeRenderer: 
    @staticmethod
    def scale_area(A):
        return A/math.pi
    
    @staticmethod
    def scale_alpha(alpha):
        return alpha*2/math.pi

    def __init__(self, cc, d_pi=0.1):
        self.alphas = [0.5*math.pi * pi_part for pi_part in np.arange(0.0, 1.0, d_pi)]
        self.cc = cc

        # All properties are defined by data above

        self.fig = plt.figure()
        self.fig.suptitle('Unified surface area from cup angle', fontsize=16)
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.ax.set_xlim([0, 1])
        self.ax.set_xlabel("Cup angle in π/2 units")
        self.ax.set_ylim([0, self.upper_limit])
        self.ax.set_ylabel("Surface area in π units")

        self.anns = []

        self.line, = self.ax.plot(self.alphas_scaled, self.As_scaled)       

        self.base_touch = self.ax.axvline( x=self.alpha_0_scaled
                                         , ymax=CoffeeRenderer.scale_area(self.cc.A(2))/self.upper_limit
                                         , c='b', ls=':'
                                         , label='base touch = {:.2f}π'.format(self.alpha_0_scaled/2)) 

        self.middle_touch = self.ax.axvline( x=self.alpha_1_scaled
                                           , ymax=CoffeeRenderer.scale_area(self.cc.A(1))/self.upper_limit
                                           , c='g', ls=':'
                                           , label='base middle = {:.2f}π'.format(self.alpha_1_scaled/2))

        self.spill_angle = self.ax.axvline( x=self.alpha_2_scaled[1]
                                          , c='k', ls=':'
                                          , label='spill angle = {:.2f}π'.format(self.alpha_2_scaled[1]/2))

        self.extremes = [self.ax.axvline(x=i) for i in range(2)]
        for line in self.extremes:
            line.set(visible=False)

        for (line, (i, (ext, ext_value))) in zip(self.extremes, enumerate(self.A_extremes_scaled)):
            line.set(xdata=[ext, ext], ydata=[0.0,ext_value/self.upper_limit], c='r', ls=':', label="extreme_{} = {:.2f}π".format(i, ext/2))
            line.set(visible=True)

        
        self.legend = self.ax.legend(loc="upper left")
        self.fig.canvas.toolbar_position = 'bottom'

    def get_update_function(self):
        def update(a = self.cc.a, D0 = self.cc.D0):
            self.cc = CoffeeCup(a=a, D0=D0)

            self.line.set_ydata(self.As_scaled)

            for line in self.extremes:
                line.set(visible=False)
                line.set(label=None)
            self
            
            for line, (i, (ext, ext_value)) in zip(self.extremes, enumerate(self.A_extremes_scaled)):
                ls = ':'
                if i == 1:
                    ls = '-.'
                line.set(xdata=[ext, ext], ydata=[0.0,ext_value/self.upper_limit], c='r', ls=ls, label="extreme{} = {:.2f}π".format("₀" if i == 0 else "₁", ext/2))
                line.set(visible=True)
            

            extreme_xs = []
            extreme_ys = []

            for ext, ext_value in self.A_extremes_scaled:
                extreme_xs.append(ext)
                extreme_ys.append(ext_value)

            A_spill_scaled, alpha_2_scaled  = self.alpha_2_scaled

            for lbl, value_x, value_y, line in zip( 
                                         ["α₀:base touch", "α₁:base middle", "α₂:spill angle"]
                                       , [self.alpha_0_scaled, self.alpha_1_scaled, alpha_2_scaled]
                                       , [CoffeeRenderer.scale_area(self.cc.A(2)), CoffeeRenderer.scale_area(self.cc.A(1)), None]
                                       , [self.base_touch, self.middle_touch, self.spill_angle]):
                
                x_data = line.get_xdata(orig=True)
                line.set_xdata(np.full_like(x_data, value_x))

                if value_y is not None:
                    line.set_ydata([0.0, value_y/self.upper_limit])
                
                line.set_label("{} = {:.2f}π".format(lbl, value_x/2))
            
            for ann in self.anns:
                ann.remove()
            del self.anns
            self.anns = []
            for i, (x, y) in enumerate(zip( [self.alpha_0_scaled, self.alpha_1_scaled, alpha_2_scaled] + extreme_xs
                                        , [CoffeeRenderer.scale_area(self.cc.A(2)), CoffeeRenderer.scale_area(self.cc.A(1)), A_spill_scaled] + extreme_ys)):
                if y is None:
                    continue
                ann = None
                if i < 2:
                    ann = self.ax.annotate("{:.2f}π".format(y), (x, y))
                elif i == 2:
                    ann = self.ax.annotate("{:.2f}π".format(y), (x, 0.2))
                elif i == 3:
                    ann = self.ax.annotate("{:.2f}π".format(y), (x, y - 0.2))
                elif i == 4:
                    ann = self.ax.annotate("{:.2f}π".format(y), (x, y - 0.2))

                self.anns.append(ann)
                

            self.legend = self.ax.legend(loc="upper left")
            self.fig.canvas.draw_idle()

        return update

    @property
    def upper_limit(self):
        # return CoffeeRenderer.scale_area(self.cc.A(2)*2)
        return 2.0

    @property
    def As(self):
        alpha0 = self.cc.alpha0
        return [math.pi/math.cos(alpha) if alpha < alpha0 else self.cc.A(self.cc.tan_alpha_reverse(math.tan(alpha))) for alpha in self.alphas]
    

    @property
    def alpha_2(self):
        alpha_2 = (None, math.pi/2.0)

        for alpha in self.alphas:
            if alpha < self.cc.alpha0:
                D = self.cc.a/math.pi + math.tan(alpha)
                if D > self.cc.D0:
                    A = math.pi/math.cos(alpha)
                    alpha_2 = (A, alpha)
                    break
            else:
                t = self.cc.tan_alpha_reverse(math.tan(alpha))
                D = self.cc.D(t)
                if D > self.cc.D0:
                    A = self.cc.A(t)
                    alpha_2 = (A, alpha)
                    break
        return alpha_2

    @property
    def alphas_scaled(self):
        return [CoffeeRenderer.scale_alpha(alpha) for alpha in self.alphas]

    @property
    def alpha_0_scaled(self):
        return CoffeeRenderer.scale_alpha(self.cc.alpha0)
    
    @property
    def alpha_1_scaled(self):
        return CoffeeRenderer.scale_alpha(self.cc.alpha1)
    
    @property
    def alpha_2_scaled(self):
        A, alpha_2 = self.alpha_2
        if A is None:
            return None, CoffeeRenderer.scale_alpha(alpha_2)
        else:
            return CoffeeRenderer.scale_area(A), CoffeeRenderer.scale_alpha(alpha_2)
    
    @property
    def As_scaled(self):
        return [CoffeeRenderer.scale_area(A) for A in self.As]

    @property
    def A_extremes_scaled(self):
        return [(CoffeeRenderer.scale_alpha(ext), CoffeeRenderer.scale_area(ext_value)) for (ext, ext_value) in self.cc.A_extremes]
    
